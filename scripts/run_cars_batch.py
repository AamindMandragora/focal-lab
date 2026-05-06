#!/usr/bin/env python3
"""Run a batch of CARS jobs described in a JSON manifest."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _require_int(name: str, value: Any) -> int:
    if value is None:
        raise ValueError(f"{name} is required")
    value_int = int(value)
    if value_int <= 0:
        raise ValueError(f"{name} must be > 0")
    return value_int


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cars-repo", type=Path, required=True)
    parser.add_argument("--grammar-file", type=Path, required=True)
    parser.add_argument("--jobs-file", type=Path, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--sample-style", type=str, default="cars")
    parser.add_argument("--default-target-samples", type=int, default=1)
    parser.add_argument("--default-n-steps", type=int, default=2000)
    parser.add_argument("--default-max-new-tokens", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    payload = json.loads(args.jobs_file.read_text())
    jobs = payload.get("jobs")
    if not isinstance(jobs, list):
        raise SystemExit(f"{args.jobs_file} must contain a 'jobs' list")

    failures: list[dict[str, Any]] = []
    for idx, job in enumerate(jobs, start=1):
        if not isinstance(job, dict):
            failures.append({"index": idx, "error": "job must be an object"})
            continue

        prompt_file = job.get("prompt_file")
        log_dir = job.get("log_dir")
        if not prompt_file or not log_dir:
            failures.append({"index": idx, "error": "job missing prompt_file or log_dir"})
            continue

        try:
            target_samples = _require_int("target_samples", job.get("target_samples", args.default_target_samples))
            n_steps = _require_int("n_steps", job.get("n_steps", args.default_n_steps))
            max_new_tokens = _require_int(
                "max_new_tokens",
                job.get("max_new_tokens", args.default_max_new_tokens),
            )
        except Exception as exc:
            failures.append({"index": idx, "prompt_file": prompt_file, "error": str(exc)})
            continue

        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_cars_task.py"),
            "--cars-repo",
            str(args.cars_repo),
            "--grammar-file",
            str(args.grammar_file),
            "--prompt-file",
            str(prompt_file),
            "--model-name",
            args.model_name,
            "--sample-style",
            args.sample_style,
            "--log-dir",
            str(log_dir),
            "--target-samples",
            str(target_samples),
            "--n-steps",
            str(n_steps),
            "--max-new-tokens",
            str(max_new_tokens),
            "--device",
            args.device,
        ]
        print(f"[cars-batch {idx}/{len(jobs)}] {' '.join(cmd)}", flush=True)
        completed = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        if completed.returncode != 0:
            failures.append({
                "index": idx,
                "prompt_file": prompt_file,
                "log_dir": log_dir,
                "return_code": completed.returncode,
            })
            break

    if failures:
        print(json.dumps({"failures": failures}, indent=2), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
