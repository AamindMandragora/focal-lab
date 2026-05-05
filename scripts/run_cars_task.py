#!/usr/bin/env python3
"""Run a single CARS grammar/prompt task with an explicit HuggingFace model ID."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cars-repo", type=Path, required=True)
    parser.add_argument("--grammar-file", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--sample-style", choices=["rs", "ars", "rsft", "cars"], default="cars")
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, default=None)
    parser.add_argument("--target-samples", type=int, default=100)
    parser.add_argument("--n-steps", type=int, default=2000)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    cars_repo = args.cars_repo.expanduser().resolve()
    grammar_file = args.grammar_file.expanduser().resolve()
    prompt_file = args.prompt_file.expanduser().resolve()
    log_dir = args.log_dir.expanduser().resolve()
    runtime_dir = (
        args.runtime_dir.expanduser().resolve()
        if args.runtime_dir is not None
        else log_dir.parent / "_cars_runtime"
    )

    if str(cars_repo) not in sys.path:
        sys.path.insert(0, str(cars_repo))

    runtime_dir.mkdir(parents=True, exist_ok=True)
    # The upstream CARS loader reads ./secrets.json. Keep that runtime file out
    # of the CARS checkout so this wrapper is non-invasive.
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
    prompt = prompt_file.read_text()
    log_dir.mkdir(parents=True, exist_ok=True)

    model = cars.lib.ConstrainedModel(args.model_name, grammar, torch_dtype=torch.bfloat16)
    runner = cars.CARS(model=model, prompt=prompt, sample_style=args.sample_style, log_dir=str(log_dir))
    runner.get_samples(
        n_samples=1,
        n_steps=args.n_steps,
        stop_after=args.target_samples,
        max_new_tokens=args.max_new_tokens,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
