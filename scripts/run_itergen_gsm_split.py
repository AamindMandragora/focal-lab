#!/usr/bin/env python3
"""Run original IterGen on the CRANE GSM-Symbolic split."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from lark import Lark
from lark.exceptions import LarkError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluations.gsm_symbolic.dataset import load_gsm_from_crane_folder
from evaluations.gsm_symbolic.grammar import build_dynamic_grammar, extract_variables_from_mapping
from project_defaults import default_crane_repo, default_gsm_source_dir, default_itergen_repo
from scripts.gsm_baseline_prompts import crane_gsm_chat_prompt, crane_gsm_text_prompt


def patch_itergen_logits_warper_compat(itergen_cls: type[Any]) -> None:
    if getattr(itergen_cls, "__vas_logits_warper_patched__", False):
        return

    def _compat_update_gen_args(self: Any, **gen_args: dict) -> None:
        from transformers.generation.logits_process import LogitsProcessorList

        self.generation_config.update(**gen_args)
        if hasattr(self.model, "_get_logits_warper"):
            self.logit_warper = self.model._get_logits_warper(self.generation_config, device=self.device)
        else:
            self.logit_warper = LogitsProcessorList()

    itergen_cls.update_gen_args = _compat_update_gen_args
    setattr(itergen_cls, "__vas_logits_warper_patched__", True)


def add_itergen_paths(itergen_repo: Path) -> None:
    for path in [
        itergen_repo,
        itergen_repo / "itergen" / "syncode" / "syncode",
    ]:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def load_indices(split_file: Path, split_name: str, limit: int | None) -> list[int]:
    manifest = json.loads(split_file.read_text())
    key = f"{split_name}_indices"
    if key not in manifest:
        available = sorted(k for k in manifest if k.endswith("_indices"))
        raise SystemExit(f"{split_file} does not contain {key}; available={available}")
    indices = list(manifest[key])
    if limit is not None and limit > 0:
        indices = indices[:limit]
    return indices


def prompt_for_example(example: dict[str, Any], model: str, crane_repo: Path) -> str | list[dict[str, str]]:
    question = example.get("question_parsed") or example.get("question") or ""
    if any(tag in model for tag in ("Instruct", "instruct", "chat", "it")):
        return crane_gsm_chat_prompt(crane_repo, question)
    return crane_gsm_text_prompt(crane_repo, question)


def clean_expression(output: str | None) -> str | None:
    if output is None:
        return None
    text = str(output).strip()
    for marker in ("<|im_end|>", "<|eot_id|>", "<|endoftext|>"):
        text = text.replace(marker, "")
    matches = re.findall(r"<<\s*([^<>]+?)\s*>>", text)
    if matches:
        return matches[-1].strip()
    if "Expression:" in text:
        text = text.rsplit("Expression:", 1)[-1].strip()
    if "The answer is" in text:
        text = re.split(r"The answer is", text, flags=re.IGNORECASE)[-1].strip()
    text = text.splitlines()[0].strip() if text else ""
    return text.rstrip(".;").strip() or None


def gsm_symbolic_equivalence(model_expr: str | None, expected_expr: str, variable_types: dict[str, Any]) -> bool:
    if model_expr is None or not expected_expr:
        return False
    import random as rng

    var_names = set(re.findall(r"\b[a-zA-Z_]\w*\b", model_expr + " " + expected_expr))
    var_names -= {"int"}
    for name in var_names:
        if name not in variable_types:
            return False
    for _ in range(200):
        env: dict[str, Any] = {}
        for var in var_names:
            vtype = str(variable_types.get(var, "int")).lower()
            if vtype == "float between 0 and 1":
                env[var] = rng.uniform(0.001, 1)
            elif vtype == "float":
                env[var] = rng.uniform(0.001, 100)
            else:
                env[var] = rng.randint(1, 100)
        try:
            val_model = eval(model_expr, {"__builtins__": {}}, {**env, "int": int})
            val_expected = eval(expected_expr, {"__builtins__": {}}, {**env, "int": int})
        except Exception:
            return False
        if abs(val_model - val_expected) > 1e-6 * max(1, abs(val_expected)):
            return False
    return True


def syntax_valid(expr: str | None, base_grammar: str, variable_types: dict[str, Any]) -> bool:
    if not expr:
        return False
    variables = extract_variables_from_mapping(variable_types)
    grammar = build_dynamic_grammar(base_grammar, variables)
    try:
        Lark(grammar, start="start", parser="lalr").parse(expr)
        return True
    except (LarkError, Exception):
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--itergen-repo", type=Path, default=default_itergen_repo())
    parser.add_argument("--crane-repo", type=Path, default=default_crane_repo())
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--split-name", choices=["train", "eval", "test"], default="eval")
    parser.add_argument("--gsm-source-dir", type=Path, default=default_gsm_source_dir())
    parser.add_argument("--grammar", type=Path, default=PROJECT_ROOT / "grammars" / "gsm.lark")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--recurrence-penalty", type=float, default=0.3)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch

    torch.manual_seed(args.seed)
    itergen_repo = args.itergen_repo.expanduser().resolve()
    crane_repo = args.crane_repo.expanduser().resolve()
    add_itergen_paths(itergen_repo)
    from itergen.main import IterGen  # type: ignore

    patch_itergen_logits_warper_compat(IterGen)
    base_grammar = args.grammar.read_text()
    indices = load_indices(args.split_file, args.split_name, args.limit)
    examples = load_gsm_from_crane_folder(args.gsm_source_dir, indices=indices)
    do_sample = args.temperature is not None

    old_cwd = Path.cwd()
    os.chdir(itergen_repo)
    try:
        iter_gen = IterGen(
            grammar=base_grammar,
            model_id=args.model,
            default_unit="syncode",
            parse_output_only=True,
            do_sample=do_sample,
            temperature=args.temperature,
            max_tokens=args.max_model_len,
            max_new_tokens=args.max_new_tokens,
            recurrence_penalty=args.recurrence_penalty,
            device=args.device,
        )

        rows: list[dict[str, Any]] = []
        correct = 0
        syntax_ok = 0
        total_tokens = 0
        start = time.time()
        for local_i, example in enumerate(examples, start=1):
            print(f"[{local_i}/{len(examples)}] GSM source_index={example.get('crane_source_index')}", flush=True)
            ex_start = time.time()
            prompt = prompt_for_example(example, args.model, crane_repo)
            raw_output = ""
            metadata: dict[str, Any] = {}
            try:
                iter_gen.start(prompt)
                out = iter_gen.forward()
                raw_output = out[0] if out else ""
                metadata = dict(iter_gen._metadata or {})
            except Exception as exc:
                metadata = {"error": str(exc)}
            expr = clean_expression(raw_output)
            variable_types = example.get("variable_types") or {}
            expected = example.get("answer_parsed") or ""
            is_syntax = syntax_valid(expr, base_grammar, variable_types)
            is_correct = gsm_symbolic_equivalence(expr, expected, variable_types)
            correct += int(is_correct)
            syntax_ok += int(is_syntax)
            token_count = int(metadata.get("total_tokens") or 0)
            total_tokens += token_count
            rows.append({
                "source_index": example.get("crane_source_index"),
                "question": example.get("question_parsed") or example.get("question", ""),
                "prediction": expr,
                "raw_output": raw_output,
                "expected": expected,
                "syntax_valid": is_syntax,
                "is_correct": is_correct,
                "token_count": token_count,
                "time_seconds": time.time() - ex_start,
                "metadata": metadata,
            })
            print(f"  -> expr: {expr} correct={is_correct} syntax={is_syntax}", flush=True)
    finally:
        os.chdir(old_cwd)

    output = {
        "config": {
            "method": "itergen",
            "dataset": "gsm",
            "itergen_repo": str(itergen_repo),
            "crane_repo": str(crane_repo),
            "prompt_source": "crane.src.prompt_templates.gsm_symbolic.cot.gsm",
            "default_unit": "syncode",
            "split_file": str(args.split_file),
            "split_name": args.split_name,
            "indices": indices,
            "model": args.model,
            "device": args.device,
            "seed": args.seed,
            "temperature": args.temperature,
            "recurrence_penalty": args.recurrence_penalty,
            "max_model_len": args.max_model_len,
            "max_new_tokens": args.max_new_tokens,
        },
        "accuracy": correct / max(1, len(rows)),
        "syntax_rate": syntax_ok / max(1, len(rows)),
        "num_correct": correct,
        "num_examples": len(rows),
        "avg_num_tokens": total_tokens / max(1, len(rows)),
        "wall_time_seconds": time.time() - start,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, default=str))
    print(f"[summary] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
