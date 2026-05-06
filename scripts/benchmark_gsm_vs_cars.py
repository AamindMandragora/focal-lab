#!/usr/bin/env python3
"""Evaluate original CARS on the CRANE GSM-Symbolic split."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
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
from project_defaults import default_gsm_source_dir


MODEL_MAP = {
    "1": "meta-llama/Llama-3.1-8B-Instruct",
    "2": "Qwen/Qwen2.5-7B-Instruct",
    "3": "Qwen/Qwen2.5-14B-Instruct",
}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def load_split_indices(split_file: Path, split_name: str, limit: int | None) -> list[int]:
    manifest = json.loads(split_file.read_text())
    key = f"{split_name}_indices"
    if key not in manifest:
        available = sorted(k for k in manifest if k.endswith("_indices"))
        raise SystemExit(f"{split_file} does not contain {key}; available={available}")
    indices = list(manifest[key])
    if limit is not None and limit > 0:
        indices = indices[:limit]
    return indices


def prompt_for_example(example: dict[str, Any]) -> str:
    question = example.get("question_parsed") or example.get("question") or ""
    variables = extract_variables_from_mapping(example.get("variable_types") or {})
    var_text = ", ".join(variables) if variables else "the variables in the problem"
    return (
        "Solve this GSM-Symbolic word problem symbolically. "
        "Output only the final arithmetic expression, with no explanation, no code fences, "
        "and no << >> delimiters. Use only these variable names: "
        f"{var_text}.\n\n"
        f"Problem: {question}\n"
        "Expression:"
    )


def write_cars_compatible_grammar(source: Path, output_path: Path) -> Path:
    """Write a CARS/llguidance-compatible GSM grammar copy.

    The repo's Lark grammar uses terminal priority syntax such as ``TYPE.4``.
    CARS' llguidance frontend rejects that extension, so the adapter removes
    priority suffixes while leaving the evaluation grammar unchanged.
    """
    text = source.read_text()
    text = re.sub(r"^([A-Z_][A-Z0-9_]*)\.\d+:", r"\1:", text, flags=re.MULTILINE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text)
    return output_path


def batch_command(args: argparse.Namespace, jobs_file: Path, grammar_file: Path) -> list[str]:
    model_name = args.model_name or MODEL_MAP[args.model_number]
    return [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_cars_batch.py"),
        "--cars-repo",
        str(args.cars_repo),
        "--grammar-file",
        str(grammar_file),
        "--jobs-file",
        str(jobs_file),
        "--model-name",
        model_name,
        "--sample-style",
        args.cars_style,
        "--default-target-samples",
        "1",
        "--default-n-steps",
        str(args.max_attempts_per_example),
        "--default-max-new-tokens",
        str(args.max_new_tokens),
    ]


def extract_first_output(log_dir: Path) -> tuple[str, dict[str, Any]]:
    candidates = sorted(log_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        return "", {"error": "missing_cars_log"}
    data = json.loads(candidates[-1].read_text())
    steps = data.get("steps") or []
    if not steps:
        return "", {"successes": data.get("successes", []), "error": "no_successful_step"}
    step = steps[0]
    output = "".join(step.get("tokens", []))
    return output, {
        "token_count": len(step.get("token_ids", [])),
        "raw_logprob": step.get("raw_logprob"),
        "constrained_logprob": step.get("cons_logprob"),
        "successes": data.get("successes", []),
    }


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
    parser.add_argument("--cars-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--split-name", choices=["train", "eval", "test"], default="eval")
    parser.add_argument("--gsm-source-dir", type=Path, default=default_gsm_source_dir())
    parser.add_argument("--grammar", type=Path, default=PROJECT_ROOT / "grammars" / "gsm.lark")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model-number", choices=sorted(MODEL_MAP), default="2")
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--cars-style", choices=["rs", "ars", "rsft", "cars"], default="cars")
    parser.add_argument("--max-attempts-per-example", type=int, default=2000)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--cuda-visible-devices", type=str, default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    indices = load_split_indices(args.split_file, args.split_name, args.limit)
    examples = load_gsm_from_crane_folder(args.gsm_source_dir, indices=indices)
    base_grammar = args.grammar.read_text()
    run_root = args.output.parent / f"{args.output.stem}_cars"
    prompt_dir = run_root / "prompts"
    log_root = run_root / "logs"
    cars_grammar = write_cars_compatible_grammar(args.grammar, run_root / "gsm_cars_grammar.lark")
    jobs: list[dict[str, Any]] = []
    commands: list[list[str]] = []
    records: list[dict[str, Any]] = []
    predictions: list[str | None] = []

    for i, example in enumerate(examples):
        source_index = int(example.get("crane_source_index", indices[i]))
        prompt_path = prompt_dir / f"gsm_{source_index:04d}.txt"
        log_dir = log_root / f"gsm_{source_index:04d}"
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(prompt_for_example(example))
        jobs.append({
            "prompt_file": str(prompt_path),
            "log_dir": str(log_dir),
            "target_samples": 1,
            "n_steps": args.max_attempts_per_example,
            "max_new_tokens": args.max_new_tokens,
        })

    jobs_file = run_root / "cars_jobs.json"
    cmd = batch_command(args, jobs_file, cars_grammar)
    commands.append(cmd)
    env = os.environ.copy()
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    start = time.time()
    if not args.dry_run:
        write_json(jobs_file, {"jobs": jobs})
        subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, check=True)

    correct = 0
    syntax_ok = 0
    total_tokens = 0
    for i, example in enumerate(examples):
        source_index = int(example.get("crane_source_index", indices[i]))
        log_dir = log_root / f"gsm_{source_index:04d}"
        raw_output = ""
        raw: dict[str, Any] = {}
        if not args.dry_run:
            raw_output, raw = extract_first_output(log_dir)
        expr = clean_expression(raw_output)
        predictions.append(expr)
        variable_types = example.get("variable_types") or {}
        expected = example.get("answer_parsed") or ""
        is_syntax = syntax_valid(expr, base_grammar, variable_types)
        is_correct = gsm_symbolic_equivalence(expr, expected, variable_types)
        correct += int(is_correct)
        syntax_ok += int(is_syntax)
        total_tokens += int(raw.get("token_count") or 0)
        records.append({
            "source_index": source_index,
            "question": example.get("question_parsed") or example.get("question", ""),
            "prediction": expr,
            "raw_output": raw_output,
            "expected": expected,
            "syntax_valid": is_syntax,
            "is_correct": is_correct,
            **raw,
        })

    output = {
        "config": {
            "method": "cars",
            "dataset": "gsm",
            "model_name": args.model_name or MODEL_MAP[args.model_number],
            "split_file": str(args.split_file),
            "split_name": args.split_name,
            "indices": indices,
            "dry_run": args.dry_run,
        },
        "commands": commands,
        "accuracy": correct / max(1, len(examples)),
        "syntax_rate": syntax_ok / max(1, len(examples)),
        "num_correct": correct,
        "num_examples": len(examples),
        "avg_tokens": total_tokens / max(1, len(examples)),
        "wall_time_seconds": time.time() - start,
        "records": records,
    }
    write_json(args.output, output)
    print(f"[summary] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
