#!/usr/bin/env python3
"""
Compare two compiled CSD strategies on the same dataset samples.

Usage:
    python compare_csd.py \\
        --dataset gsm_symbolic \\
        --csd1 path/to/baseline_csd \\
        --csd2 path/to/new_csd \\
        --label1 "CRANE" --label2 "Synthesized" \\
        --sample-size 10 \\
        --model Qwen/Qwen2.5-Coder-7B-Instruct \\
        --device cuda
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_comparison(args):
    from synthesis.evaluator import Evaluator

    print(f"\n{'='*60}")
    print(f"CSD Comparison: {args.dataset}")
    print(f"  {args.label1}: {args.csd1}")
    print(f"  {args.label2}: {args.csd2}")
    print(f"  Sample size: {args.sample_size}, Model: {args.model}")
    print(f"{'='*60}\n")

    evaluator = Evaluator(
        dataset_name=args.dataset,
        model_name=args.model,
        device=args.device,
        sample_size=args.sample_size,
        max_steps=args.max_steps,
    )

    results = {}
    for label, csd_path in [(args.label1, Path(args.csd1)), (args.label2, Path(args.csd2))]:
        print(f"Evaluating {label}...")
        # Reset dataset so both use seed=42
        evaluator._dataset = None
        evaluator._env = None

        result = evaluator.evaluate_sample(csd_path, sample_size=args.sample_size)
        results[label] = result

        print(f"  Accuracy:    {result.accuracy:.1%} ({result.num_correct}/{result.num_examples})")
        print(f"  Format rate: {result.format_rate:.1%}")
        print(f"  Syntax rate: {result.syntax_rate:.1%}")
        print(f"  Time:        {result.total_time_seconds:.1f}s")
        if result.error:
            print(f"  ERROR: {result.error}")
        print()

    # Summary comparison table
    print(f"\n{'='*60}")
    print(f"{'Metric':<20} {args.label1:>15} {args.label2:>15} {'Delta':>10}")
    print(f"{'-'*60}")
    for metric, key in [("Accuracy", "accuracy"), ("Format Rate", "format_rate"), ("Syntax Rate", "syntax_rate")]:
        v1 = getattr(results[args.label1], key)
        v2 = getattr(results[args.label2], key)
        delta = v2 - v1
        arrow = "+" if delta > 0 else ""
        print(f"{metric:<20} {v1:>14.1%} {v2:>14.1%} {arrow}{delta:>9.1%}")
    print(f"{'='*60}")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({
                label: result.to_dict()
                for label, result in results.items()
            }, f, indent=2)
        print(f"\nResults saved to {output_path}")

    # Show per-example details
    if args.verbose:
        for label, result in results.items():
            print(f"\n--- {label} per-example ---")
            for i, s in enumerate(result.sample_outputs):
                status = "✓" if s.get("is_correct") else "✗"
                fmt = "F" if s.get("is_valid_format") else "-"
                syn = f"{s.get('syntax_rate', 0):.0%}"
                q = s.get("question", "")[:60]
                print(f"  [{status}][{fmt}][{syn}] {q}")
                if not s.get("is_correct"):
                    actual_str = s.get("actual") or "?"
                    print(f"         Expected: {s.get('expected')} | Got: {actual_str[:40]}")


def main():
    parser = argparse.ArgumentParser(description="Compare two CSD strategies")
    parser.add_argument("--dataset", required=True, choices=["gsm_symbolic", "folio"])
    parser.add_argument("--csd1", required=True, help="Path to first compiled CSD directory")
    parser.add_argument("--csd2", required=True, help="Path to second compiled CSD directory")
    parser.add_argument("--label1", default="CRANE", help="Label for first CSD")
    parser.add_argument("--label2", default="Synthesized", help="Label for second CSD")
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--output", help="Save JSON results to this path")
    parser.add_argument("--verbose", action="store_true", help="Show per-example details")
    args = parser.parse_args()
    run_comparison(args)


if __name__ == "__main__":
    main()
