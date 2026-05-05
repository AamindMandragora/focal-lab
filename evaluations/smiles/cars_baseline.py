"""Native SMILES baselines (RS/ARS/CARS) inspired by arXiv:2510.01902."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch

from evaluations.smiles.dataset import get_smiles_task
from evaluations.smiles.metrics import evaluate_smiles_output
from synthesis.evaluator import Evaluator


@dataclass
class _TrieNode:
    terminal: bool = False
    children: dict[int, "_TrieNode"] = field(default_factory=dict)


def _trie_add(root: _TrieNode, token_ids: Sequence[int]) -> None:
    node = root
    for tok in token_ids:
        node = node.children.setdefault(int(tok), _TrieNode())
    node.terminal = True


def _trie_walk(root: _TrieNode, prefix: Sequence[int]) -> Optional[_TrieNode]:
    node = root
    for tok in prefix:
        node = node.children.get(int(tok))
        if node is None:
            return None
        if node.terminal:
            return node
    return node


def _to_dafny_prefix(_dafny, token_strs: Sequence[str]):
    return _dafny.SeqWithoutIsStrInference([_dafny.Seq(tok) for tok in token_strs])


def _model_probs_for_prefix(lm, _dafny, token_strs: Sequence[str]) -> torch.Tensor:
    lm.GenerateLogits(_to_dafny_prefix(_dafny, token_strs))
    logits = lm._full_logits if getattr(lm, "_full_logits", None) is not None else lm._logits_tensor
    probs = torch.softmax(logits.float(), dim=0)
    if torch.isnan(probs).any() or probs.sum().item() <= 0:
        probs = torch.zeros_like(logits, dtype=torch.float32)
        probs[int(torch.argmax(logits).item())] = 1.0
    return probs


def _invalid_prefix_token_ids(
    parser,
    _dafny,
    token_ids: Sequence[int],
    lm,
) -> Optional[tuple[int, ...]]:
    token_strs = [lm._token_str_from_id(int(tid)) for tid in token_ids]
    if not token_strs:
        return tuple()

    full = _to_dafny_prefix(_dafny, token_strs)
    try:
        if parser.IsCompletePrefix(full):
            return None
    except Exception:
        pass

    for i in range(1, len(token_strs) + 1):
        pref = _to_dafny_prefix(_dafny, token_strs[:i])
        if not parser.IsValidPrefix(pref):
            return tuple(int(t) for t in token_ids[:i])

    return tuple(int(t) for t in token_ids)


def _sample_token_with_cars(
    *,
    lm,
    _dafny,
    prefix_ids: Sequence[int],
    prefix_token_strs: Sequence[str],
    forbidden_root: _TrieNode,
    probs_cache: dict[tuple[int, ...], torch.Tensor],
    survival_cache: dict[tuple[int, ...], float],
) -> int:
    prefix_key = tuple(int(x) for x in prefix_ids)

    def probs_for(prefix_ids_local: tuple[int, ...], token_strs_local: Sequence[str]) -> torch.Tensor:
        cached = probs_cache.get(prefix_ids_local)
        if cached is not None:
            return cached
        p = _model_probs_for_prefix(lm, _dafny, token_strs_local)
        probs_cache[prefix_ids_local] = p
        return p

    def survival(prefix_ids_local: tuple[int, ...], token_strs_local: Sequence[str], node: Optional[_TrieNode]) -> float:
        if node is None:
            return 1.0
        if node.terminal:
            return 0.0
        cached = survival_cache.get(prefix_ids_local)
        if cached is not None:
            return cached

        probs_local = probs_for(prefix_ids_local, token_strs_local)
        val = 1.0
        for tok_id, child in node.children.items():
            p_tok = float(probs_local[int(tok_id)].item()) if int(tok_id) < probs_local.numel() else 0.0
            if p_tok <= 0.0:
                continue
            if child.terminal:
                child_survival = 0.0
            else:
                tok_str = lm._token_str_from_id(int(tok_id))
                child_survival = survival(
                    prefix_ids_local + (int(tok_id),),
                    list(token_strs_local) + [tok_str],
                    child,
                )
            val -= p_tok * (1.0 - child_survival)

        val = max(0.0, min(1.0, val))
        survival_cache[prefix_ids_local] = val
        return val

    probs = probs_for(prefix_key, prefix_token_strs)
    node = _trie_walk(forbidden_root, prefix_ids)
    if node is None or node.terminal or not node.children:
        return int(torch.multinomial(probs, num_samples=1).item())

    adjusted = probs.clone()
    for tok_id, child in node.children.items():
        tok_id_i = int(tok_id)
        if tok_id_i >= adjusted.numel():
            continue
        if child.terminal:
            adjusted[tok_id_i] = 0.0
        else:
            tok_str = lm._token_str_from_id(tok_id_i)
            child_survival = survival(
                prefix_key + (tok_id_i,),
                list(prefix_token_strs) + [tok_str],
                child,
            )
            adjusted[tok_id_i] = adjusted[tok_id_i] * float(child_survival)

    total = float(adjusted.sum().item())
    if total <= 0.0 or torch.isnan(adjusted).any():
        return int(torch.multinomial(probs, num_samples=1).item())
    adjusted = adjusted / total
    return int(torch.multinomial(adjusted, num_samples=1).item())


def _generate_one(
    *,
    env: Dict[str, Any],
    parser,
    prompt: str,
    max_steps: int,
    style: str,
    forbidden_root: _TrieNode,
) -> tuple[str, list[int], float]:
    _dafny = env["_dafny"]
    lm = env["lm"]
    eos_id = lm.tokenizer.eos_token_id
    if eos_id is None:
        eos_token = lm.tokenizer.eos_token or "<|endoftext|>"
        eos_ids = lm.tokenizer.encode(eos_token, add_special_tokens=False)
        eos_id = int(eos_ids[0]) if eos_ids else -1

    lm.instruction_text = lm.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )

    token_ids: list[int] = []
    token_strs: list[str] = []
    started = time.time()

    probs_cache: dict[tuple[int, ...], torch.Tensor] = {}
    survival_cache: dict[tuple[int, ...], float] = {}
    for _ in range(max_steps):
        prefix_seq = _to_dafny_prefix(_dafny, token_strs)
        lm.GenerateLogits(prefix_seq)
        if style == "cars":
            next_id = _sample_token_with_cars(
                lm=lm,
                _dafny=_dafny,
                prefix_ids=token_ids,
                prefix_token_strs=token_strs,
                forbidden_root=forbidden_root,
                probs_cache=probs_cache,
                survival_cache=survival_cache,
            )
        else:
            probs = lm._full_logits if getattr(lm, "_full_logits", None) is not None else lm._logits_tensor
            probs = torch.softmax(probs.float(), dim=0)
            if torch.isnan(probs).any() or probs.sum().item() <= 0:
                next_id = int(torch.argmax(lm._full_logits if getattr(lm, "_full_logits", None) is not None else lm._logits_tensor).item())
            else:
                next_id = int(torch.multinomial(probs, num_samples=1).item())

        if int(next_id) == int(eos_id):
            break
        token_ids.append(int(next_id))
        token_strs.append(lm._token_str_from_id(int(next_id)))

    return "".join(token_strs), token_ids, time.time() - started


def run_native_smiles_baseline(
    *,
    compiled_module: str,
    classes: Sequence[str],
    model_name: str,
    backend: str,
    device: str,
    max_steps: int,
    target_samples: int,
    max_attempts: int,
    style: str,
) -> list[dict[str, Any]]:
    """
    Run native RS/ARS/CARS over SMILES tasks.

    style:
      - "rs": plain rejection sampling (no remembered invalids)
      - "ars": adaptive rejection with full-sequence invalid memory
      - "cars": adaptive rejection with shortest-invalid-prefix memory
    """
    if style not in {"rs", "ars", "cars"}:
        raise ValueError(f"Unsupported native baseline style: {style}")

    evaluator = Evaluator(
        dataset_name="smiles",
        model_name=model_name,
        backend=backend,
        device=device,
        sample_size=1,
        max_steps=max_steps,
    )

    env = evaluator._setup_environment(compiled_module_path=Path(compiled_module))
    summaries: list[dict[str, Any]] = []

    for class_name in classes:
        task = get_smiles_task(class_name)
        parser = evaluator._build_smiles_dynamic_parser(env, task)
        if parser is None:
            parser = env["parser"]

        forbidden_root = _TrieNode()
        unique_valid: set[str] = set()
        records: list[dict[str, Any]] = []
        started = time.time()

        for attempt in range(1, max_attempts + 1):
            out_text, token_ids, dt = _generate_one(
                env=env,
                parser=parser,
                prompt=task["prompt"],
                max_steps=max_steps,
                style=style,
                forbidden_root=forbidden_root,
            )

            eval_row = evaluate_smiles_output(
                class_name=class_name,
                output=out_text,
                grammar_text=task["grammar_text"],
                prompt_exemplars=task["prompt_exemplars"],
                require_rdkit=True,
            )

            records.append(
                {
                    "attempt": attempt,
                    "output": out_text,
                    "time_seconds": dt,
                    "token_count": len(token_ids),
                    **eval_row,
                }
            )

            if eval_row.get("unique_valid_candidate"):
                unique_valid.add(str(eval_row.get("smiles") or ""))
            else:
                if style in {"ars", "cars"}:
                    invalid_pref = _invalid_prefix_token_ids(parser, env["_dafny"], token_ids, env["lm"])
                    if invalid_pref:
                        if style == "ars":
                            _trie_add(forbidden_root, tuple(int(t) for t in token_ids))
                        else:
                            _trie_add(forbidden_root, invalid_pref)

            if len(unique_valid) >= target_samples:
                break

        syntax_count = sum(1 for r in records if r.get("syntax_valid"))
        membership_count = sum(1 for r in records if r.get("class_membership"))
        summaries.append(
            {
                "class_name": class_name,
                "style": style,
                "attempt_count": len(records),
                "unique_valid_count": len(unique_valid),
                "reached_target": len(unique_valid) >= target_samples,
                "syntax_rate": syntax_count / max(1, len(records)),
                "accuracy": membership_count / max(1, len(records)),
                "wall_time": time.time() - started,
                "records": records,
            }
        )

    return summaries
