#!/usr/bin/env python3
"""Run the unconstrained baseline on GSM-Symbolic, Spider, or SMILES.

This runner does not compile or execute any generated Dafny module. It imports
the original CRANE repository's plain HuggingFace generation wrapper
(`BaseLM(mode="original")`) and scores those unconstrained outputs with this
repo's split-specific dataset scorers.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from synthesis.evaluator import Evaluator


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
        repo / "syncode",
        repo / "syncode" / "syncode",
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
    for module_name in ("syncode", "parsers"):
        if module_name in sys.modules:
            del sys.modules[module_name]


def task_name_for_original_lm(dataset: str) -> str:
    if dataset == "gsm_symbolic":
        return "gsm_symbolic"
    if dataset == "spider":
        return "spider"
    if dataset == "smiles":
        return "smiles"
    raise ValueError(f"Unsupported dataset: {dataset}")


def build_original_unconstrained_lm(args: argparse.Namespace, *, dataset: str):
    add_crane_paths(args.crane_repo)
    from models.base_model import BaseLM  # type: ignore

    lm = BaseLM(
        model_name=args.eval_model,
        mode="original",
        grammar="json",
        max_tokens=args.eval_max_steps,
        temperature=0.0,
        device=args.crane_device,
        task=task_name_for_original_lm(dataset),
        start_symbol="<<",
        start_in_grammar=True,
        end_symbol=">>",
        end_in_grammar=True,
    )
    return lm


def generate_unconstrained(lm: Any, prompt: str | list[dict[str, str]]) -> tuple[str, int, float]:
    batch = lm(
        [{"prompt": prompt}],
        prompt_key="prompt",
        response_key="llm_response",
        info_key="response_info",
    )
    row = batch[0] if batch else {}
    info = row.get("response_info") or {}
    return (
        str(row.get("llm_response") or ""),
        int(info.get("tokens") or 0),
        float(info.get("time") or 0.0),
    )


def clean_gsm_expression(output: str) -> str | None:
    text = str(output or "").strip()
    for marker in ("<|im_end|>", "<|eot_id|>", "<|endoftext|>"):
        text = text.replace(marker, "")
    matches = re.findall(r"<<\s*([^<>]+?)\s*>>", text)
    if matches:
        return matches[-1].strip()
    if "Expression:" in text:
        text = text.rsplit("Expression:", 1)[-1].strip()
    if "The final answer is" in text:
        text = re.split(r"The final answer is", text, flags=re.IGNORECASE)[-1].strip()
    if "The answer is" in text:
        text = re.split(r"The answer is", text, flags=re.IGNORECASE)[-1].strip()
    text = text.splitlines()[0].strip() if text else ""
    return text.rstrip(".;").strip() or None


def score_output(
    evaluator: Evaluator,
    *,
    dataset: str,
    example: dict[str, Any],
    output_text: str,
) -> tuple[bool, bool, dict[str, Any]]:
    expected = evaluator._get_expected_answer(example)
    scored_output = evaluator._truncate_gsm_output(output_text) if dataset == "gsm_symbolic" else output_text

    if dataset == "gsm_symbolic":
        actual = clean_gsm_expression(scored_output)
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
            "answer_source": "gsm_unconstrained_text_or_delimiter",
            "syntax_segments": [{"text": text, "valid": valid} for text, valid in segments],
        }

    if dataset == "spider":
        actual = evaluator._extract_answer_spider(scored_output)
        is_correct = evaluator._exec_match_spider(actual, expected, example)
        syntax_valid = bool(actual)
        return is_correct, syntax_valid, {
            "expected": expected,
            "actual": actual,
            "answer_source": "spider_unconstrained_sql_extractor",
        }

    if dataset == "smiles":
        from evaluations.smiles.metrics import evaluate_smiles_output

        smiles_eval = evaluate_smiles_output(
            example.get("class_name", ""),
            scored_output,
            example.get("grammar_text", evaluator._get_grammar_text()),
            example.get("prompt_exemplars", []),
        )
        return bool(smiles_eval.get("valid_class_membership")), bool(smiles_eval.get("syntax_valid")), {
            "expected": expected,
            "actual": smiles_eval.get("smiles"),
            "answer_source": "smiles_unconstrained_extractor",
            "smiles_eval": smiles_eval,
        }

    raise ValueError(f"Unsupported dataset: {dataset}")


def run_dataset(
    args: argparse.Namespace,
    *,
    dataset: str,
    lm: Any,
    sample_size: int,
    smiles_class: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    evaluator = Evaluator(
        dataset_name=dataset,
        model_name=args.eval_model,
        backend=args.eval_backend,
        device=args.device,
        sample_size=sample_size,
        max_steps=args.eval_max_steps,
        step_token_budget=args.eval_step_token_budget,
        gsm_source_dir=args.gsm_source_dir,
        gsm_split_file=args.gsm_split_file,
        gsm_split_name=args.gsm_split_name,
        spider_split_file=args.spider_split_file,
        spider_split_name=args.spider_split_name,
        smiles_classes=smiles_class,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_max_model_len=args.vllm_max_model_len,
    )
    examples = evaluator._load_dataset_sample()

    rows: list[dict[str, Any]] = []
    num_correct = 0
    syntax_count = 0
    accuracy_denominator = 0
    total_tokens = 0
    start = time.time()

    for i, example in enumerate(examples, start=1):
        print(f"[unconstrained:{dataset}] {i}/{len(examples)}", flush=True)
        prompt = evaluator._format_prompt(example)
        try:
            output_text, token_count, gen_time = generate_unconstrained(lm, prompt)
            is_correct, syntax_valid, score_meta = score_output(
                evaluator,
                dataset=dataset,
                example=example,
                output_text=output_text,
            )
            accuracy_applicable = syntax_valid if dataset == "smiles" else True
            if accuracy_applicable:
                accuracy_denominator += 1
            num_correct += int(is_correct)
            syntax_count += int(syntax_valid)
            total_tokens += int(token_count or 0)
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
    payload = {
        "success": True,
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
        "invalid_outputs_excluded_from_accuracy": (
            num_examples - accuracy_denominator if dataset == "smiles" else 0
        ),
        "avg_num_tokens": total_tokens / max(1, num_examples),
        "wall_time_seconds": time.time() - start,
        "sample_outputs": rows,
    }
    return payload, list(examples)


def run_smiles_class_target(
    args: argparse.Namespace,
    *,
    lm: Any,
    class_name: str,
    target_samples: int,
) -> dict[str, Any]:
    evaluator = Evaluator(
        dataset_name="smiles",
        model_name=args.eval_model,
        backend=args.eval_backend,
        device=args.device,
        sample_size=1,
        max_steps=args.eval_max_steps,
        step_token_budget=args.eval_step_token_budget,
        smiles_classes=class_name,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_max_model_len=args.vllm_max_model_len,
    )
    examples = evaluator._load_dataset_sample()
    example = examples[0]
    prompt = evaluator._format_prompt(example)

    rows: list[dict[str, Any]] = []
    unique_valid: set[str] = set()
    total_tokens = 0
    attempts = 0
    start = time.time()
    while attempts < args.smiles_max_attempts and len(unique_valid) < target_samples:
        attempts += 1
        try:
            output_text, token_count, gen_time = generate_unconstrained(lm, prompt)
            is_correct, syntax_valid, score_meta = score_output(
                evaluator,
                dataset="smiles",
                example=example,
                output_text=output_text,
            )
            smiles_eval = score_meta.get("smiles_eval", {})
            if smiles_eval.get("unique_valid_candidate"):
                unique_valid.add(smiles_eval.get("smiles", ""))
            total_tokens += int(token_count or 0)
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
                f"  [unconstrained-smiles:{class_name}] attempts={attempts} "
                f"unique_valid={len(unique_valid)}/{target_samples}",
                flush=True,
            )

    syntax_count = sum(1 for row in rows if row.get("is_syntax_valid"))
    valid_membership_count = sum(1 for row in rows if row.get("is_correct"))
    membership_count_all = sum(
        1
        for row in rows
        if (row.get("smiles_eval") or {}).get("class_membership")
    )
    return {
        "class_name": class_name,
        "target_samples": target_samples,
        "max_attempts": args.smiles_max_attempts,
        "attempt_count": attempts,
        "success_count": sum(1 for row in rows if row.get("success")),
        "unique_valid_count": len(unique_valid),
        "reached_target": len(unique_valid) >= target_samples,
        "num_examples": len(rows),
        "num_correct": valid_membership_count,
        "accuracy": valid_membership_count / syntax_count if syntax_count else None,
        "syntax_rate": syntax_count / max(1, len(rows)),
        "accuracy_denominator": syntax_count,
        "accuracy_definition": "class_membership_among_syntax_valid_molecules",
        "invalid_outputs_excluded_from_accuracy": len(rows) - syntax_count,
        "membership_rate_all_attempts": membership_count_all / max(1, len(rows)),
        "avg_num_tokens": total_tokens / max(1, len(rows)),
        "wall_time_seconds": time.time() - start,
        "sample_outputs": rows,
    }


def add_spider_official_scores(payload: dict[str, Any], examples: list[dict[str, Any]]) -> None:
    from evaluations.sql_spider.dataset import default_db_dir, default_tables_json
    from evaluations.sql_spider.executor import execute_accuracy

    predictions = [
        str(row.get("actual") or "")
        for row in payload.get("sample_outputs", [])
    ]
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
    validity_values = [row.get("validity") for row in rows if "validity" in row]
    if validity_values:
        payload["syntax_rate"] = sum(1 for value in validity_values if value) / len(validity_values)
        payload["syntax_definition"] = "Spider executor validity over generated SQL predictions"


def aggregate_smiles(class_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    total_examples = sum(int(p.get("num_examples", 0) or 0) for p in class_payloads)
    total_accuracy_denominator = sum(int(p.get("accuracy_denominator", 0) or 0) for p in class_payloads)
    total_correct = sum(int(p.get("num_correct", 0) or 0) for p in class_payloads)
    syntax_pass = sum(
        float(p.get("syntax_rate", 0.0) or 0.0) * int(p.get("num_examples", 0) or 0)
        for p in class_payloads
    )
    invalid_excluded = sum(
        int(p.get("invalid_outputs_excluded_from_accuracy", 0) or 0)
        for p in class_payloads
    )
    return {
        "classes": class_payloads,
        "num_examples": total_examples,
        "num_correct": total_correct,
        "accuracy": total_correct / max(1, total_accuracy_denominator),
        "syntax_rate": syntax_pass / max(1, total_examples),
        "accuracy_denominator": total_accuracy_denominator,
        "accuracy_definition": "class_membership_among_syntax_valid_molecules",
        "invalid_outputs_excluded_from_accuracy": invalid_excluded,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["gsm", "gsm_symbolic", "spider", "sql", "smiles"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "generated-csd")
    parser.add_argument("--crane-repo", type=Path, default=Path("/home/aadivyar/CRANE"))
    parser.add_argument("--crane-device", default="cuda:0")
    parser.add_argument("--eval-model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--eval-backend", choices=["huggingface", "vllm"], default="vllm")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--eval-max-steps", type=int, default=512)
    parser.add_argument("--eval-step-token-budget", type=int, default=1)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--vllm-max-model-len", type=int, default=4096)
    parser.add_argument("--gsm-source-dir", type=Path, default=Path("/home/aadivyar/CRANE/src/gsm_symbolic"))
    parser.add_argument("--gsm-split-file", type=Path, default=None)
    parser.add_argument("--gsm-split-name", choices=["train", "eval", "test"], default="eval")
    parser.add_argument("--spider-split-file", type=Path, default=None)
    parser.add_argument("--spider-split-name", choices=["train", "test", "eval"], default="test")
    parser.add_argument("--smiles-classes", default="acrylates,chain_extenders,isocyanates")
    parser.add_argument("--smiles-max-attempts", type=int, default=2000)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dataset = normalize_dataset(args.dataset)
    start = time.time()

    if args.dry_run:
        payload = {
            "config": vars(args),
            "dataset": dataset,
            "dry_run": True,
            "framework_source": "original_crane_repo.BaseLM(mode=original)",
            "crane_repo": str(args.crane_repo),
        }
        write_json(args.output, payload)
        print(f"[dry-run] would write {args.output}")
        return 0

    lm = build_original_unconstrained_lm(args, dataset=dataset)
    if dataset == "smiles":
        class_payloads: list[dict[str, Any]] = []
        for class_name in normalize_smiles_classes(args.smiles_classes):
            print(f"[unconstrained-smiles:{class_name}] sample_size={args.sample_size}", flush=True)
            class_payload = run_smiles_class_target(
                args,
                lm=lm,
                class_name=class_name,
                target_samples=args.sample_size,
            )
            class_payloads.append(class_payload)
        result = aggregate_smiles(class_payloads)
    else:
        result, examples = run_dataset(
            args,
            dataset=dataset,
            lm=lm,
            sample_size=args.sample_size,
        )
        if dataset == "spider":
            add_spider_official_scores(result, examples)

    payload = {
        "config": {
            **vars(args),
            "dataset": dataset,
            "method": "unconstrained",
            "framework_source": "original_crane_repo.BaseLM(mode=original)",
            "crane_repo": str(args.crane_repo),
        },
        **result,
        "wall_time_seconds": time.time() - start,
    }
    write_json(args.output, payload)
    print(f"[summary] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
