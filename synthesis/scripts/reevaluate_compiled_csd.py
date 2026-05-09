#!/usr/bin/env python3
"""Re-run GSM (or other) evaluation on an already compiled GeneratedCSD.py path."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from synthesis.evaluate.evaluator import Evaluator


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "compiled_generated_csd",
        type=Path,
        help="Path to GeneratedCSD.py inside the compiled output folder",
    )
    p.add_argument("--dataset", default="gsm_symbolic")
    p.add_argument("--eval-model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    p.add_argument("--sample-size", type=int, default=15)
    p.add_argument("--max-steps", type=int, default=900)
    p.add_argument("--step-token-budget", type=int, default=1)
    p.add_argument("--vllm-max-model-len", type=int, default=4096)
    args = p.parse_args()

    if os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") is None:
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    compiled = args.compiled_generated_csd.resolve()
    if not compiled.is_file():
        raise SystemExit(f"Not a file: {compiled}")

    ev = Evaluator(
        dataset_name=args.dataset,
        model_name=args.eval_model,
        backend="vllm",
        device="cuda",
        sample_size=args.sample_size,
        max_steps=args.max_steps,
        step_token_budget=args.step_token_budget,
        vllm_max_model_len=args.vllm_max_model_len,
        vllm_enforce_eager=True,
    )
    try:
        res = ev.evaluate_sample(compiled, sample_size=args.sample_size)
    finally:
        ev.unload_runtime()

    print(f"accuracy: {res.accuracy:.4f}")
    print(f"syntax_rate: {res.syntax_rate:.4f}")
    print(f"num_correct: {res.num_correct} / {res.num_examples}")
    print(f"contains_delimiters: {res.contains_delimiters}")
    syn = sum(1 for s in res.sample_outputs if s.get("is_syntax_valid"))
    print(f"per_example_syntax_pass: {syn} / {len(res.sample_outputs)}")


if __name__ == "__main__":
    main()
