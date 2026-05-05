#!/usr/bin/env python3
"""Run CARS over many prompt files with one model load."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

PAPER_CHAT_MODELS = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    "Qwen/Qwen2.5-Math-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "Qwen/Qwen2.5-Coder-14B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
]


def load_jobs(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        jobs = payload.get("jobs", [])
    else:
        jobs = payload
    if not isinstance(jobs, list):
        raise ValueError(f"{path} must contain a list or a dict with jobs")
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cars-repo", type=Path, required=True)
    parser.add_argument("--grammar-file", type=Path, required=True)
    parser.add_argument("--jobs-file", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--sample-style", choices=["rs", "ars", "rsft", "cars"], default="cars")
    parser.add_argument("--runtime-dir", type=Path, default=None)
    parser.add_argument("--default-target-samples", type=int, default=1)
    parser.add_argument("--default-n-steps", type=int, default=2000)
    parser.add_argument("--default-max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    cars_repo = args.cars_repo.expanduser().resolve()
    grammar_file = args.grammar_file.expanduser().resolve()
    jobs_file = args.jobs_file.expanduser().resolve()
    runtime_dir = (
        args.runtime_dir.expanduser().resolve()
        if args.runtime_dir is not None
        else jobs_file.parent / "_cars_runtime"
    )

    if str(cars_repo) not in sys.path:
        sys.path.insert(0, str(cars_repo))

    runtime_dir.mkdir(parents=True, exist_ok=True)
    secrets_path = runtime_dir / "secrets.json"
    if not secrets_path.exists():
        secrets_path.write_text(json.dumps({"HF_TOKEN": os.environ.get("HF_TOKEN", "your_token")}))
    os.chdir(runtime_dir)

    import cars  # type: ignore
    import cars.lib  # type: ignore

    for model_id in PAPER_CHAT_MODELS:
        if model_id not in cars.lib.ConstrainedModel.HF_CHAT_MODELS:
            cars.lib.ConstrainedModel.HF_CHAT_MODELS.append(model_id)
    if args.model_name not in cars.lib.ConstrainedModel.HF_CHAT_MODELS:
        cars.lib.ConstrainedModel.HF_CHAT_MODELS.append(args.model_name)

    grammar = grammar_file.read_text()
    model = cars.lib.ConstrainedModel(args.model_name, grammar, torch_dtype=torch.bfloat16)

    jobs = load_jobs(jobs_file)
    summary: list[dict[str, Any]] = []
    for i, job in enumerate(jobs, start=1):
        prompt_path = Path(job["prompt_file"]).expanduser().resolve()
        log_dir = Path(job["log_dir"]).expanduser().resolve()
        prompt = prompt_path.read_text()
        target_samples = int(job.get("target_samples", args.default_target_samples))
        n_steps = int(job.get("n_steps", args.default_n_steps))
        max_new_tokens = int(job.get("max_new_tokens", args.default_max_new_tokens))
        print(f"[cars-batch {i}/{len(jobs)}] prompt={prompt_path} log_dir={log_dir}", flush=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        runner = cars.CARS(
            model=model,
            prompt=prompt,
            sample_style=job.get("sample_style", args.sample_style),
            log_dir=str(log_dir),
        )
        runner.get_samples(
            n_samples=1,
            n_steps=n_steps,
            stop_after=target_samples,
            max_new_tokens=max_new_tokens,
        )
        summary.append({
            "prompt_file": str(prompt_path),
            "log_dir": str(log_dir),
            "target_samples": target_samples,
            "n_steps": n_steps,
            "max_new_tokens": max_new_tokens,
        })

    summary_path = jobs_file.with_suffix(".summary.json")
    summary_path.write_text(json.dumps({"jobs": summary}, indent=2))
    print(f"[summary] wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
