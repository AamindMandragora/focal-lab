#!/usr/bin/env python3
"""Held-out evaluation script for CSD strategies.

Evaluates a compiled CSD strategy (GeneratedCSD.py) on a GSM-Symbolic sample
using vLLM. Saves aggregate metrics AND per-example outputs so results can be
queried later (e.g. "which examples did winner3 get right that CRANE missed?").

Usage:
    python scripts/held_out_eval.py \
        --module outputs/.../synth_lottery_9/GeneratedCSD.py \
        --label winner9 \
        --gpu 1 \
        --sample-size 50 \
        --seed 456

Output: outputs/generated-csd/runs/held_out_eval/{label}.json containing:
    - aggregate metrics (accuracy, syntax_rate, etc.)
    - sample_outputs[] with per-example question/expected/actual/is_correct
"""
import argparse
import json
import os
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--module", required=True, help="Path to GeneratedCSD.py")
    p.add_argument("--label", required=True, help="Label for output file (e.g. 'winner3', 'crane')")
    p.add_argument("--gpu", type=int, required=True, help="CUDA device id")
    p.add_argument("--sample-size", type=int, default=50)
    p.add_argument("--seed", type=int, default=456)
    p.add_argument(
        "--eval-random",
        type=lambda v: str(v).lower() in ("1", "true", "yes", "y", "t"),
        default=True,
        help="If True (default), sample randomly with --seed. If False, first-N deterministic."
    )
    p.add_argument("--model", default="Qwen/Qwen2.5-Coder-14B-Instruct")
    p.add_argument("--dataset", default="gsm_symbolic")
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--max-seconds-per-example", type=float, default=120.0)
    p.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--vllm-max-model-len", type=int, default=8192)
    p.add_argument(
        "--output-dir",
        default="/home/aadivyar/csd-generation/outputs/generated-csd/runs/held_out_eval",
    )
    args = p.parse_args()

    # Pin GPU before importing torch/vllm
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    sys.path.insert(0, "/home/aadivyar/csd-generation")
    from synthesis.evaluator import Evaluator

    print(f"[held_out_eval] label={args.label} gpu={args.gpu} n={args.sample_size} seed={args.seed}")
    print(f"[held_out_eval] module={args.module}")

    evaluator = Evaluator(
        dataset_name=args.dataset,
        model_name=args.model,
        backend="vllm",
        device="cuda",
        sample_size=args.sample_size,
        max_steps=args.max_steps,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_max_model_len=args.vllm_max_model_len,
        vllm_enforce_eager=True,
        sample_seed=args.seed,
        random_sample=args.eval_random,
        max_seconds_per_example=args.max_seconds_per_example,
    )

    result = evaluator.evaluate_sample(Path(args.module))

    output = {
        "label": args.label,
        "module_path": str(args.module),
        "sample_size": args.sample_size,
        "seed": args.seed,
        "model": args.model,
        "dataset": args.dataset,
        # Aggregate metrics
        "success": result.success,
        "accuracy": result.accuracy,
        "syntax_rate": result.syntax_rate,
        "contains_delimiters": result.contains_delimiters,
        "num_correct": result.num_correct,
        "num_examples": result.num_examples,
        "total_time_seconds": result.total_time_seconds,
        "max_sample_time_seconds": result.max_sample_time_seconds,
        "error": result.error,
        # Per-example outputs: question, expected, actual, is_correct, time_seconds, etc.
        "sample_outputs": result.sample_outputs,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, f"{args.label}.json")
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(
        f"[held_out_eval] {args.label}: "
        f"acc={result.accuracy:.1%} ({result.num_correct}/{result.num_examples}) "
        f"syntax={result.syntax_rate:.1%} time={result.total_time_seconds:.1f}s"
    )
    print(f"[held_out_eval] saved to {out_file}")


if __name__ == "__main__":
    main()
