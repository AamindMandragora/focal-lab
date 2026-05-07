#!/usr/bin/env python3
"""Evaluate original CARS on an explicit Spider split."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.native_libs import ensure_env_lib_first

ensure_env_lib_first()

from evaluations.sql_spider.dataset import load_spider
from evaluations.sql_spider.executor import _clean_sql, execute_accuracy


MODEL_MAP = {
    "1": "meta-llama/Llama-3.1-8B-Instruct",
    "2": "Qwen/Qwen2.5-7B-Instruct",
    "3": "Qwen/Qwen2.5-14B-Instruct",
}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def load_split_indices(split_file: Path | None, split_name: str) -> list[int] | None:
    if split_file is None:
        return None
    manifest = json.loads(split_file.read_text())
    key = f"{split_name}_indices"
    if key not in manifest:
        available = sorted(k for k in manifest if k.endswith("_indices"))
        raise SystemExit(f"{split_file} does not contain {key}; available={available}")
    return list(manifest[key])


def command_for_example(
    args: argparse.Namespace,
    *,
    source_index: int,
    log_dir: Path,
) -> list[str]:
    cars_repo = args.cars_repo.expanduser().resolve()
    grammar = cars_repo / "datasets" / "spider" / "grammar.lark"
    prompt = cars_repo / "datasets" / "spider" / f"instance_{source_index:04d}.txt"
    if not grammar.exists():
        raise FileNotFoundError(grammar)
    if not prompt.exists():
        raise FileNotFoundError(prompt)
    model_name = args.model_name or MODEL_MAP[args.model_number]
    return [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_cars_task.py"),
        "--cars-repo",
        str(cars_repo),
        "--grammar-file",
        str(grammar),
        "--prompt-file",
        str(prompt),
        "--model-name",
        model_name,
        "--sample-style",
        args.cars_style,
        "--log-dir",
        str(log_dir),
        "--target-samples",
        "1",
        "--n-steps",
        str(args.max_attempts_per_example),
        "--max-new-tokens",
        str(args.max_new_tokens),
    ]


def batch_command(args: argparse.Namespace, jobs_file: Path) -> list[str]:
    cars_repo = args.cars_repo.expanduser().resolve()
    grammar = cars_repo / "datasets" / "spider" / "grammar.lark"
    model_name = args.model_name or MODEL_MAP[args.model_number]
    return [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_cars_batch.py"),
        "--cars-repo",
        str(cars_repo),
        "--grammar-file",
        str(grammar),
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


def extract_first_prediction(log_dir: Path) -> tuple[str, dict[str, Any]]:
    candidates = sorted(log_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        return "", {"error": "missing_cars_log"}
    data = json.loads(candidates[-1].read_text())
    steps = data.get("steps") or []
    if not steps:
        return "", {"successes": data.get("successes", []), "error": "no_successful_step"}
    step = steps[0]
    output = "".join(step.get("tokens", []))
    return _clean_sql(output), {
        "raw_output": output,
        "token_count": len(step.get("token_ids", [])),
        "raw_logprob": step.get("raw_logprob"),
        "constrained_logprob": step.get("cons_logprob"),
        "successes": data.get("successes", []),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cars-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--split-name", choices=["train", "test", "eval"], default="test")
    parser.add_argument("--source", choices=["auto", "hf", "local"], default="auto")
    parser.add_argument("--spider-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model-number", choices=sorted(MODEL_MAP), default="2")
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--cars-style", choices=["rs", "ars", "rsft", "cars"], default="cars")
    parser.add_argument("--max-attempts-per-example", type=int, default=500)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--cuda-visible-devices", type=str, default="")
    parser.add_argument("--etype", choices=["exec", "match", "all"], default="exec")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    indices = load_split_indices(args.split_file, args.split_name)
    examples = load_spider(
        source=args.source,
        spider_dir=args.spider_dir,
        indices=indices,
        limit=args.limit,
    )
    if args.limit is not None:
        examples = examples[: args.limit]

    output_dir = args.output.parent / f"{args.output.stem}_cars_logs"
    commands: list[list[str]] = []
    predictions: list[str] = []
    records: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    env = os.environ.copy()
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    start = time.time()
    for ordinal, example in enumerate(examples):
        source_index = int(example.get("spider_source_index", ordinal))
        log_dir = output_dir / f"instance_{source_index:04d}"
        cmd = command_for_example(args, source_index=source_index, log_dir=log_dir)
        commands.append(cmd)
        jobs.append({
            "prompt_file": str(args.cars_repo.expanduser().resolve() / "datasets" / "spider" / f"instance_{source_index:04d}.txt"),
            "log_dir": str(log_dir),
            "target_samples": 1,
            "n_steps": args.max_attempts_per_example,
            "max_new_tokens": args.max_new_tokens,
        })
        print(f"[cars-spider {ordinal + 1}/{len(examples)}] source_index={source_index}")
        print(" ".join(cmd))
        if args.dry_run:
            predictions.append("")
            records.append({"source_index": source_index, "command": cmd})
            continue

    jobs_file = args.output.parent / f"{args.output.stem}_cars_jobs.json"
    if not args.dry_run:
        write_json(jobs_file, {"jobs": jobs})
        batch_cmd = batch_command(args, jobs_file)
        commands = [batch_cmd]
        subprocess.run(batch_cmd, cwd=str(PROJECT_ROOT), env=env, check=True)

    if not args.dry_run:
        for ordinal, example in enumerate(examples):
            source_index = int(example.get("spider_source_index", ordinal))
            log_dir = output_dir / f"instance_{source_index:04d}"
            pred, raw = extract_first_prediction(log_dir)
            predictions.append(pred)
            records.append({
                "source_index": source_index,
                "db_id": example.get("db_id", ""),
                "question": example.get("question", ""),
                "prediction": pred,
                **raw,
            })

    payload: dict[str, Any] = {
        "config": {
            "method": "cars",
            "dataset": "spider",
            "model_name": args.model_name or MODEL_MAP[args.model_number],
            "split_file": str(args.split_file) if args.split_file else None,
            "split_name": args.split_name if args.split_file else None,
            "num_examples": len(examples),
            "dry_run": args.dry_run,
        },
        "commands": commands,
        "records": records,
        "wall_time_seconds": time.time() - start,
    }
    if not args.dry_run:
        scores, error_types, rows = execute_accuracy(
            predictions=predictions,
            examples=examples,
            etype=args.etype,
        )
        payload["scores"] = scores
        payload["error_types"] = error_types
        payload["rows"] = rows

    write_json(args.output, payload)
    print(f"[summary] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
