#!/usr/bin/env python3
"""Evaluate the original CRANE framework on the experiment datasets.

This runner does not compile or execute our generated Dafny strategies. It
imports CRANE's original Python decoder from `--crane-repo` and feeds it the
dataset grammar/prompts, then scores the outputs with this repo's dataset
scorers so the split and metrics match the rest of the matrix.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.native_libs import ensure_env_lib_first

ensure_env_lib_first()

from synthesis.evaluator import Evaluator
from project_defaults import default_crane_repo, default_gsm_source_dir
from scripts.gsm_baseline_prompts import crane_gsm_chat_prompt


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def normalize_dataset(raw: str) -> str:
    if raw == "gsm":
        return "gsm_symbolic"
    if raw == "sql":
        return "spider"
    return raw


def normalize_smiles_classes(raw: str) -> list[str]:
    from evaluations.smiles.dataset import SMILES_CLASSES

    classes = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = sorted(set(classes) - set(SMILES_CLASSES))
    if unknown:
        raise SystemExit(f"Unknown SMILES class(es): {unknown}")
    return classes


def add_crane_paths(crane_repo: Path) -> None:
    repo = crane_repo.expanduser().resolve()
    paths = [
        repo / "src",
        repo / "src" / "itergen",
        repo / "src" / "itergen" / "iter_syncode",
        repo / "upstream-uiuc" / "src",
    ]
    for path in paths:
        if path.exists():
            path_str = str(path)
            if path_str in sys.path:
                sys.path.remove(path_str)
            sys.path.insert(0, path_str)


def load_crane_class(
    crane_repo: Path,
    *,
    stable_mode: bool = False,
    decoder_source: str = "auto",
):
    add_crane_paths(crane_repo)

    def _patch_logits_warper_compat(module: Any) -> None:
        itergen_cls = getattr(module, "IterGen", None)
        if itergen_cls is None:
            return
        if getattr(itergen_cls, "_vas_logits_warper_patched", False):
            return

        def _compat_update_gen_args(self, **gen_args: dict) -> None:
            self.cursors = [0 for _ in range(self.num_outputs)]
            self.generation_config.update(**gen_args)
            default_int_fields = {
                "num_return_sequences": 1,
                "num_beams": 1,
                "num_beam_groups": 1,
            }
            for field_name, default_value in default_int_fields.items():
                if getattr(self.generation_config, field_name, None) is None:
                    setattr(self.generation_config, field_name, default_value)
            if getattr(self.generation_config, "do_sample", None) is None:
                self.generation_config.do_sample = False
            get_warper = getattr(self.model, "_get_logits_warper", None)
            if callable(get_warper):
                self.logit_warper = get_warper(self.generation_config, device=self.device)
            else:
                self.logit_warper = lambda _token_ids, scores: scores

        itergen_cls.update_gen_args = _compat_update_gen_args
        itergen_cls._vas_logits_warper_patched = True

    def _patch_dfa_warning_spam() -> None:
        try:
            import crane.iter_syncode.dfa_mask_store as dfa_mask_store  # type: ignore
        except Exception:
            return
        lookup_cls = getattr(dfa_mask_store, "LookupTable", None)
        if lookup_cls is None or getattr(lookup_cls, "_vas_warn_patch_applied", False):
            return

        original = lookup_cls.incomplete_case_lookup

        def _incomplete_case_lookup_once(self, dfa_state):  # type: ignore[no-untyped-def]
            assert isinstance(dfa_state, dfa_mask_store.DFAState)
            if self._mode == "grammar_mask":
                return self._overapprox_lookup[dfa_state]
            if self._mode == "grammar_strict":
                if dfa_state in self._exact_lookup:
                    return self._exact_lookup[dfa_state]
                warned = getattr(self, "_vas_warned_dfa_states", None)
                if warned is None:
                    warned = set()
                    setattr(self, "_vas_warned_dfa_states", warned)
                if dfa_state not in warned:
                    warned.add(dfa_state)
                    print(
                        f"Warning: Exact lookup not found for {dfa_state} in the DFA mask store. Falling back to overapprox.",
                        flush=True,
                    )
                return self._overapprox_lookup[dfa_state]
            return original(self, dfa_state)

        lookup_cls.incomplete_case_lookup = _incomplete_case_lookup_once
        lookup_cls._vas_warn_patch_applied = True

    if not stable_mode:
        _patch_dfa_warning_spam()

    itergen_error: Exception | None = None
    if decoder_source in {"auto", "itergen"}:
        try:
            from itergen.main import AdaptiveConstrainedDecoder as CraneDecoder  # type: ignore
            import itergen.main as itergen_main  # type: ignore

            _patch_logits_warper_compat(itergen_main)
            return CraneDecoder, "itergen.main.AdaptiveConstrainedDecoder"
        except Exception as exc:
            itergen_error = exc
            if decoder_source == "itergen":
                raise RuntimeError(f"Requested --decoder-source itergen, but import failed: {exc}") from exc

    if decoder_source in {"auto", "crane"}:
        from crane.main import CRANE as CraneDecoder  # type: ignore
        import crane.main as crane_main  # type: ignore

        _patch_logits_warper_compat(crane_main)
        source = "crane.main.CRANE"
        if itergen_error is not None and decoder_source == "auto":
            source = f"{source} (itergen_import_failed: {itergen_error})"
        return CraneDecoder, source

    raise ValueError(f"Unsupported decoder_source={decoder_source!r}")


def _inline_common_lark_imports(grammar_text: str) -> str:
    replacements = {
        r"^\s*%import\s+common\.CNAME\s*->\s*VARIABLE\s*$": 'VARIABLE: /[a-zA-Z_][a-zA-Z0-9_]*/',
        r"^\s*%import\s+common\.NUMBER\s*$": r"NUMBER: /-?\d+(\.\d+)?/",
        r"^\s*%import\s+common\.WS_INLINE\s*$": r"WS_INLINE: /[ \t\f]+/",
        r"^\s*%import\s+common\.WS\s*$": r"WS: /[ \t\f\r\n]+/",
    }
    result = grammar_text
    for pattern, replacement in replacements.items():
        result = re.sub(pattern, lambda _m, rep=replacement: rep, result, flags=re.MULTILINE)
    return result


def grammar_for_dataset(dataset: str, smiles_class: str | None = None) -> str:
    if dataset == "gsm_symbolic":
        return _inline_common_lark_imports((PROJECT_ROOT / "grammars" / "gsm_crane.lark").read_text())
    if dataset == "spider":
        return _inline_common_lark_imports((PROJECT_ROOT / "grammars" / "sql.lark").read_text())
    if dataset == "smiles":
        from evaluations.smiles.dataset import get_smiles_task

        if not smiles_class:
            raise ValueError("smiles_class is required for SMILES grammar")
        return _inline_common_lark_imports(str(get_smiles_task(smiles_class)["grammar_text"]))
    raise ValueError(f"Unsupported dataset: {dataset}")


def prompt_for_crane(
    evaluator: Evaluator,
    dataset: str,
    example: dict[str, Any],
    *,
    stable_mode: bool = False,
    crane_repo: Path | None = None,
) -> str | list[dict[str, str]]:
    if dataset == "gsm_symbolic":
        question = example.get("question_parsed") or example.get("question", "")
        if crane_repo is not None:
            return crane_gsm_chat_prompt(crane_repo, question)
        from evaluations.gsm_symbolic.prompts import reasoning_with_symbolic_expr_prompt

        return reasoning_with_symbolic_expr_prompt(question)

    prompt = evaluator._format_prompt(example)
    if dataset == "spider":
        marker_instruction = (
            "Begin your answer with << and then output only the SQL query. "
            "Do not include explanation or code fences."
        )
        if isinstance(prompt, list):
            prompt = [dict(m) for m in prompt]
            prompt[-1]["content"] = f"{prompt[-1]['content']}\n\n{marker_instruction}\n<<"
            return prompt
        return f"{prompt}\n\n{marker_instruction}\n<<"
    if dataset == "smiles":
        marker_instruction = "Begin with << and then output only one SMILES string."
        if isinstance(prompt, list):
            prompt = [dict(m) for m in prompt]
            prompt[-1]["content"] = f"{prompt[-1]['content']}\n\n{marker_instruction}\n<<"
            return prompt
        return f"{prompt}\n\n{marker_instruction}\n<<"
    return prompt


def clean_gsm_expression(output: str, *, prioritize_final_answer_span: bool = True) -> str | None:
    text = str(output or "").strip()
    for marker in ("<|im_end|>", "<|eot_id|>", "<|endoftext|>"):
        text = text.replace(marker, "")
    if prioritize_final_answer_span:
        final_answer_matches = re.findall(
            r"the final answer is\s*<<\s*([^<>]+?)\s*>>",
            text,
            flags=re.IGNORECASE,
        )
        if final_answer_matches:
            return final_answer_matches[-1].strip()
    matches = re.findall(r"<<\s*([^<>]+?)\s*>>", text)
    if matches:
        return matches[-1].strip()
    if "The final answer is" in text:
        text = re.split(r"The final answer is", text, flags=re.IGNORECASE)[-1].strip()
    if "The answer is" in text:
        text = re.split(r"The answer is", text, flags=re.IGNORECASE)[-1].strip()
    text = text.splitlines()[0].strip() if text else ""
    return text.rstrip(".;").strip() or None


def strip_marker(output: str) -> str:
    text = str(output or "")
    if "<<" in text:
        text = text.split("<<", 1)[1]
    if ">>" in text:
        text = text.split(">>", 1)[0]
    return text.strip()


def score_output(
    evaluator: Evaluator,
    *,
    dataset: str,
    example: dict[str, Any],
    output_text: str,
    stable_mode: bool = False,
) -> tuple[bool, bool, dict[str, Any]]:
    expected = evaluator._get_expected_answer(example)
    if dataset == "gsm_symbolic":
        scored_output = evaluator._truncate_gsm_output(output_text)
        actual = clean_gsm_expression(scored_output, prioritize_final_answer_span=not stable_mode)
        variable_types = example.get("variable_types") or {}
        if isinstance(variable_types, str):
            try:
                variable_types = eval(variable_types)
            except Exception:
                variable_types = {}
        is_correct = evaluator._gsm_symbolic_equivalence(actual, expected, variable_types) if expected else False
        all_valid_syntax, segments = evaluator._check_syntax_validity(scored_output, example=example)
        syntax_valid = bool(segments) and all_valid_syntax
        return is_correct, syntax_valid, {
            "expected": expected,
            "actual": actual,
            "scored_output": scored_output,
            "answer_source": "crane_original_gsm_delimiter",
            "syntax_segments": [{"text": text, "valid": valid} for text, valid in segments],
        }

    if dataset == "spider":
        from evaluations.sql_spider.executor import _clean_sql

        actual = _clean_sql(evaluator._extract_answer_spider(output_text))
        is_correct = evaluator._exec_match_spider(actual, expected, example)
        return is_correct, bool(actual), {
            "expected": expected,
            "actual": actual,
            "scored_output": actual,
            "answer_source": "crane_original_sql_extractor",
        }

    if dataset == "smiles":
        from evaluations.smiles.metrics import evaluate_smiles_output

        scored_output = strip_marker(output_text)
        smiles_eval = evaluate_smiles_output(
            example.get("class_name", ""),
            scored_output,
            example.get("grammar_text", evaluator._get_grammar_text()),
            example.get("prompt_exemplars", []),
        )
        return bool(smiles_eval.get("valid_class_membership")), bool(smiles_eval.get("syntax_valid")), {
            "expected": expected,
            "actual": smiles_eval.get("smiles"),
            "scored_output": scored_output,
            "answer_source": "crane_original_smiles_extractor",
            "smiles_eval": smiles_eval,
        }

    raise ValueError(f"Unsupported dataset: {dataset}")


def build_crane_decoder(args: argparse.Namespace, *, dataset: str, smiles_class: str | None = None):
    CraneDecoder, source = load_crane_class(
        args.crane_repo,
        stable_mode=args.stable_mode,
        decoder_source=args.decoder_source,
    )
    grammar = grammar_for_dataset(dataset, smiles_class=smiles_class)
    start_symbol = "<<" if dataset in {"gsm_symbolic", "spider", "smiles"} else args.start_symbol
    start_in_grammar = dataset == "gsm_symbolic"
    end_symbol = ">>" if dataset == "gsm_symbolic" else None
    end_in_grammar = dataset == "gsm_symbolic"
    stop_strings = [">>"] if dataset == "gsm_symbolic" and args.stable_mode else []
    decoder = CraneDecoder(
        grammar=grammar,
        model_id=args.eval_model,
        parse_output_only=True,
        recurrence_penalty=args.recurrence_penalty,
        stop_strings=stop_strings,
        device=args.crane_device,
        max_tokens=args.crane_max_model_len,
        max_new_tokens=args.eval_max_steps,
        start_symbol=start_symbol,
        start_in_grammar=start_in_grammar,
        end_symbol=end_symbol,
        end_in_grammar=end_in_grammar,
    )
    return decoder, source, {
        "grammar_source": "dataset_lark_text",
        "prompt_source": (
            "crane.src.prompt_templates.gsm_symbolic.cot.gsm"
            if dataset == "gsm_symbolic"
            else "evaluator"
        ),
        "start_symbol": start_symbol,
        "start_in_grammar": start_in_grammar,
        "end_symbol": end_symbol,
        "end_in_grammar": end_in_grammar,
        "stop_strings": stop_strings,
    }


def run_single_generation(
    decoder: Any,
    evaluator: Evaluator,
    *,
    dataset: str,
    example: dict[str, Any],
    stable_mode: bool = False,
    crane_repo: Path | None = None,
) -> tuple[str, int, float]:
    prompt = prompt_for_crane(
        evaluator,
        dataset,
        example,
        stable_mode=stable_mode,
        crane_repo=crane_repo,
    )
    kwargs: dict[str, Any] = {}
    if dataset == "gsm_symbolic":
        from evaluations.gsm_symbolic.grammar import extract_variables_from_mapping

        variable_types = example.get("variable_types") or {}
        if isinstance(variable_types, str):
            try:
                variable_types = eval(variable_types)
            except Exception:
                variable_types = {}
        valid_vars = extract_variables_from_mapping(variable_types) if isinstance(variable_types, dict) else []
        if valid_vars:
            kwargs["valid_vars"] = valid_vars
    start = time.time()
    try:
        decoder.start(prompt, **kwargs)
    except TypeError:
        decoder.start(prompt)
    output = decoder.forward()
    text = output[0] if isinstance(output, list) else str(output)
    token_count = int((getattr(decoder, "_metadata", {}) or {}).get("total_tokens") or 0)
    return text, token_count, time.time() - start


def run_dataset(
    args: argparse.Namespace,
    *,
    dataset: str,
    sample_size: int,
    smiles_class: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    evaluator = Evaluator(
        dataset_name=dataset,
        model_name=args.eval_model,
        backend="huggingface",
        device=args.crane_device,
        sample_size=sample_size,
        max_steps=args.eval_max_steps,
        gsm_source_dir=args.gsm_source_dir,
        gsm_split_file=args.gsm_split_file,
        gsm_split_name=args.gsm_split_name,
        spider_split_file=args.spider_split_file,
        spider_split_name=args.spider_split_name,
        smiles_classes=smiles_class,
    )
    examples = evaluator._load_dataset_sample()
    decoder, source, decoder_config = build_crane_decoder(
        args,
        dataset=dataset,
        smiles_class=smiles_class,
    )

    rows: list[dict[str, Any]] = []
    num_correct = 0
    syntax_count = 0
    accuracy_denominator = 0
    total_tokens = 0
    start = time.time()
    for i, example in enumerate(examples, start=1):
        print(f"[crane-original:{dataset}] {i}/{len(examples)}", flush=True)
        try:
            output_text, token_count, gen_time = run_single_generation(
                decoder,
                evaluator,
                dataset=dataset,
                example=example,
                stable_mode=args.stable_mode,
                crane_repo=args.crane_repo,
            )
            is_correct, syntax_valid, score_meta = score_output(
                evaluator,
                dataset=dataset,
                example=example,
                output_text=output_text,
                stable_mode=args.stable_mode,
            )
            accuracy_applicable = syntax_valid if dataset == "smiles" else True
            if accuracy_applicable:
                accuracy_denominator += 1
            num_correct += int(is_correct)
            syntax_count += int(syntax_valid)
            total_tokens += token_count
            rows.append({
                "success": True,
                "full_output": output_text,
                "token_count": token_count,
                "time_seconds": gen_time,
                "is_correct": is_correct,
                "is_syntax_valid": syntax_valid,
                "accuracy_applicable": accuracy_applicable,
                **score_meta,
            })
        except Exception as exc:
            rows.append({
                "success": False,
                "full_output": "",
                "token_count": 0,
                "time_seconds": 0.0,
                "is_correct": False,
                "is_syntax_valid": False,
                "accuracy_applicable": dataset != "smiles",
                "error": str(exc),
            })
            if dataset != "smiles":
                accuracy_denominator += 1
    num_examples = len(rows)
    return {
        "success": True,
        "crane_framework_source": source,
        "decoder_config": decoder_config,
        "accuracy": num_correct / max(1, accuracy_denominator),
        "syntax_rate": syntax_count / max(1, num_examples),
        "num_examples": num_examples,
        "num_correct": num_correct,
        "accuracy_denominator": accuracy_denominator,
        "accuracy_definition": (
            "class_membership_among_syntax_valid_molecules"
            if dataset == "smiles"
            else "correct_examples_over_all_examples"
        ),
        "invalid_outputs_excluded_from_accuracy": num_examples - accuracy_denominator if dataset == "smiles" else 0,
        "avg_num_tokens": total_tokens / max(1, num_examples),
        "wall_time_seconds": time.time() - start,
        "sample_outputs": rows,
    }, list(examples)


def run_smiles_target(
    args: argparse.Namespace,
    *,
    class_name: str,
    target_samples: int,
) -> dict[str, Any]:
    evaluator = Evaluator(
        dataset_name="smiles",
        model_name=args.eval_model,
        backend="huggingface",
        device=args.crane_device,
        sample_size=1,
        max_steps=args.eval_max_steps,
        smiles_classes=class_name,
    )
    examples = evaluator._load_dataset_sample()
    example = examples[0]
    decoder, source, decoder_config = build_crane_decoder(args, dataset="smiles", smiles_class=class_name)

    rows: list[dict[str, Any]] = []
    unique_valid: set[str] = set()
    total_tokens = 0
    attempts = 0
    start = time.time()
    while attempts < args.smiles_max_attempts and len(unique_valid) < target_samples:
        attempts += 1
        try:
            output_text, token_count, gen_time = run_single_generation(
                decoder,
                evaluator,
                dataset="smiles",
                example=example,
                stable_mode=args.stable_mode,
                crane_repo=args.crane_repo,
            )
            is_correct, syntax_valid, score_meta = score_output(
                evaluator,
                dataset="smiles",
                example=example,
                output_text=output_text,
                stable_mode=args.stable_mode,
            )
            smiles_eval = score_meta.get("smiles_eval", {})
            if smiles_eval.get("unique_valid_candidate"):
                unique_valid.add(smiles_eval.get("smiles", ""))
            total_tokens += token_count
            rows.append({
                "success": True,
                "full_output": output_text,
                "token_count": token_count,
                "time_seconds": gen_time,
                "is_correct": is_correct,
                "is_syntax_valid": syntax_valid,
                "accuracy_applicable": syntax_valid,
                **score_meta,
            })
        except Exception as exc:
            rows.append({
                "success": False,
                "full_output": "",
                "token_count": 0,
                "time_seconds": 0.0,
                "is_correct": False,
                "is_syntax_valid": False,
                "accuracy_applicable": False,
                "error": str(exc),
            })
        if attempts % 10 == 0 or len(unique_valid) >= target_samples:
            print(
                f"  [crane-original-smiles:{class_name}] attempts={attempts} "
                f"unique_valid={len(unique_valid)}/{target_samples}",
                flush=True,
            )

    syntax_count = sum(1 for row in rows if row.get("is_syntax_valid"))
    valid_membership_count = sum(1 for row in rows if row.get("is_correct"))
    membership_count_all = sum(1 for row in rows if (row.get("smiles_eval") or {}).get("class_membership"))
    return {
        "class_name": class_name,
        "crane_framework_source": source,
        "decoder_config": decoder_config,
        "target_samples": target_samples,
        "max_attempts": args.smiles_max_attempts,
        "attempt_count": attempts,
        "success_count": sum(1 for row in rows if row.get("success")),
        "unique_valid_count": len(unique_valid),
        "reached_target": len(unique_valid) >= target_samples,
        "num_examples": len(rows),
        "syntax_rate": syntax_count / max(1, len(rows)),
        "accuracy": valid_membership_count / syntax_count if syntax_count else None,
        "accuracy_definition": "class_membership_among_syntax_valid_molecules",
        "num_correct": valid_membership_count,
        "accuracy_num_correct": valid_membership_count,
        "accuracy_denominator": syntax_count,
        "invalid_outputs_excluded_from_accuracy": len(rows) - syntax_count,
        "membership_rate_all_attempts": membership_count_all / max(1, len(rows)),
        "avg_num_tokens": total_tokens / max(1, len(rows)),
        "wall_time": time.time() - start,
        "records": rows,
    }


def add_spider_official_scores(payload: dict[str, Any], examples: list[dict[str, Any]]) -> None:
    from evaluations.sql_spider.dataset import default_db_dir, default_tables_json
    from evaluations.sql_spider.executor import _clean_sql, execute_accuracy

    predictions = [_clean_sql(str(row.get("actual") or "")) for row in payload.get("sample_outputs", [])]
    scores, error_types, rows = execute_accuracy(
        predictions=predictions,
        examples=examples,
        db_dir=default_db_dir(),
        tables_json=default_tables_json(),
        etype="exec",
    )
    payload["scores"] = scores
    payload["error_types"] = error_types
    payload["rows"] = rows
    payload["all_exec_accuracy"] = float(scores.get("all", {}).get("exec", 0.0) or 0.0)
    payload["accuracy"] = payload["all_exec_accuracy"]
    payload["num_correct"] = sum(1 for row in rows if row.get("exec") is True)
    payload["accuracy_denominator"] = len(rows)
    payload["accuracy_definition"] = "Spider official execution accuracy"
    validity_values = [row.get("validity") for row in rows if "validity" in row]
    if validity_values:
        payload["syntax_rate"] = sum(1 for value in validity_values if value == "Valid") / len(validity_values)
        payload["syntax_definition"] = "Spider executor validity over generated SQL predictions"


def aggregate_smiles(class_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    total_examples = sum(int(p.get("num_examples", 0) or 0) for p in class_payloads)
    total_syntax_examples = sum(int(p.get("accuracy_denominator", 0) or 0) for p in class_payloads)
    total_correct = sum(int(p.get("num_correct", 0) or 0) for p in class_payloads)
    total_invalid_excluded = sum(
        int(p.get("invalid_outputs_excluded_from_accuracy", 0) or 0)
        for p in class_payloads
    )
    syntax_pass = sum(
        float(p.get("syntax_rate", 0.0) or 0.0) * int(p.get("num_examples", 0) or 0)
        for p in class_payloads
    )
    return {
        "classes": class_payloads,
        "num_examples": total_examples,
        "num_correct": total_correct,
        "accuracy": total_correct / max(1, total_syntax_examples),
        "syntax_rate": syntax_pass / max(1, total_examples),
        "accuracy_denominator": total_syntax_examples,
        "accuracy_definition": "class_membership_among_syntax_valid_molecules",
        "invalid_outputs_excluded_from_accuracy": total_invalid_excluded,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["gsm", "gsm_symbolic", "spider", "sql", "smiles"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "generated-csd")
    parser.add_argument("--crane-repo", type=Path, default=default_crane_repo())
    parser.add_argument("--eval-model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--eval-backend", choices=["huggingface", "vllm"], default="huggingface",
                        help="Accepted for matrix compatibility; original CRANE uses its HuggingFace backend.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--crane-device", default=None)
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--eval-max-steps", type=int, default=512)
    parser.add_argument("--eval-step-token-budget", type=int, default=1, help="Accepted for CLI compatibility; not used by original CRANE.")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.75, help="Accepted for CLI compatibility; not used by original CRANE.")
    parser.add_argument("--vllm-max-model-len", type=int, default=4096, help="Accepted for CLI compatibility; not used by original CRANE.")
    parser.add_argument("--crane-max-model-len", type=int, default=8192)
    parser.add_argument("--recurrence-penalty", type=float, default=1.0)
    parser.add_argument("--start-symbol", default="<<")
    parser.add_argument("--gsm-source-dir", type=Path, default=default_gsm_source_dir())
    parser.add_argument("--gsm-split-file", type=Path, default=None)
    parser.add_argument("--gsm-split-name", choices=["train", "eval", "test"], default="eval")
    parser.add_argument("--spider-split-file", type=Path, default=None)
    parser.add_argument("--spider-split-name", choices=["train", "test", "eval"], default="test")
    parser.add_argument("--smiles-classes", default="acrylates,chain_extenders,isocyanates")
    parser.add_argument("--smiles-max-attempts", type=int, default=500)
    parser.add_argument(
        "--decoder-source",
        choices=["auto", "crane", "itergen"],
        default="auto",
        help="Choose which CRANE decoder class to use.",
    )
    parser.add_argument(
        "--stable-mode",
        action="store_true",
        help="Use historical prompt/extraction behavior while keeping only minimal crash-compat patches.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.crane_device = args.crane_device or args.device
    if args.stable_mode and args.decoder_source == "auto":
        args.decoder_source = "crane"

    dataset = normalize_dataset(args.dataset)
    start = time.time()

    if args.dry_run:
        CraneDecoder, source = load_crane_class(
            args.crane_repo,
            stable_mode=args.stable_mode,
            decoder_source=args.decoder_source,
        )
        payload = {
            "config": {
                **vars(args),
                "dataset": dataset,
                "method": "crane",
                "framework": "original_crane_repo",
                "crane_framework_source": source,
                "crane_class": getattr(CraneDecoder, "__name__", str(CraneDecoder)),
            },
            "dry_run": True,
        }
        write_json(args.output, payload)
        print(f"[dry-run] original CRANE framework source={source}")
        print(f"[dry-run] would write {args.output}")
        return 0

    if dataset == "smiles":
        class_payloads = [
            run_smiles_target(args, class_name=class_name, target_samples=args.sample_size)
            for class_name in normalize_smiles_classes(args.smiles_classes)
        ]
        result = aggregate_smiles(class_payloads)
    else:
        result, examples = run_dataset(args, dataset=dataset, sample_size=args.sample_size)
        if dataset == "spider":
            add_spider_official_scores(result, examples)

    payload = {
        "config": {
            **vars(args),
            "dataset": dataset,
            "method": "crane",
            "framework": "original_crane_repo",
        },
        **result,
        "wall_time_seconds": time.time() - start,
    }
    write_json(args.output, payload)
    print(f"[summary] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
