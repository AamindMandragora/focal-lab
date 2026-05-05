#!/usr/bin/env python3
"""
SMILES evaluation CLI.

Runs CSD or an unconstrained baseline on the SMILES benchmark and reports
answer accuracy, constrained-format rate, syntax validity, and runtime stats.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from evaluations.smiles.dataset import load_smiles
from evaluations.smiles.environment import (
    setup_dafny_environment,
    verify_critical_tokens,
)
from evaluations.smiles.generation import run_crane_csd, run_unconstrained
from evaluations.smiles.metrics import SmilesMetrics, evaluate_smiles_output
from synthesis.evaluator import Evaluator


PROJECT_ROOT = Path(__file__).parent.parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate constrained molecular generation with CSD")
    ap.add_argument("--run-dir", type=Path, required=True, help="Path to compiled CSD run directory")
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct", help="Model ID")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=100, help="Max examples to evaluate")
    ap.add_argument("--max-steps", type=int, default=256, help="Max generation steps")
    ap.add_argument("--grammar", type=Path, default=PROJECT_ROOT / "grammars" / "smiles.lark")
    ap.add_argument("--random-sample", action="store_true", help="Randomly sample examples")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--unconstrained", action="store_true", help="Run unconstrained baseline instead of CSD")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--load-in-8bit", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    examples = load_smiles(limit=args.limit, random_sample=args.random_sample, seed=args.seed)
    metrics = SmilesMetrics()
    helper = Evaluator(dataset_name="smiles")

    print("Setting up Dafny environment...")
    dafny_env = setup_dafny_environment(
        run_dir=args.run_dir,
        model_name=args.model,
        device=args.device,
        grammar_file=args.grammar,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
    )
    verify_critical_tokens(dafny_env["tokenizer"])
    print("Model loaded. Starting evaluation...\n")

    eval_start_time = time.time()
    for index, example in enumerate(examples, start=1):
        prompt = example["prompt"]
        class_name = example["class_name"]
        print(f"[{index}/{len(examples)}] {class_name}: {example['question'][:80]}", flush=True)

        if args.unconstrained:
            output_text, token_count, time_seconds = run_unconstrained(
                dafny_env,
                prompt,
                args.max_steps,
                debug=args.verbose,
            )
            constrained_segments = []
        else:
            output_text, token_count, time_seconds, constrained_segments, _ = run_crane_csd(
                dafny_env,
                prompt,
                args.max_steps,
                args.grammar,
                debug_delimiters=args.verbose,
            )

        smiles_eval = evaluate_smiles_output(
            class_name,
            output_text,
            example["grammar_text"],
            example["prompt_exemplars"],
        )
        actual = smiles_eval["smiles"]
        is_correct = smiles_eval["unique_valid_candidate"]
        contains_delimiters = helper._contains_delimiters(output_text)
        all_valid_syntax = smiles_eval["syntax_valid"]
        syntax_segments = [(actual, all_valid_syntax)] if actual else []

        metrics.update(
            is_correct=is_correct,
            contains_delimiters=contains_delimiters,
            token_count=token_count,
            time_seconds=time_seconds,
            constrained_segments=syntax_segments,
        )

        avg_time = metrics.total_time / max(1, metrics.n)
        remaining = len(examples) - metrics.n
        eta_seconds = avg_time * remaining
        eta_str = f"{eta_seconds:.0f}s" if eta_seconds < 60 else f"{eta_seconds / 60:.1f}m"
        syntax_str = "yes" if (contains_delimiters and all_valid_syntax) else "no"
        print(
            f"  -> Answer: {actual or '(none)'} | Correct: {is_correct} | Syntax: {syntax_str} | "
            f"Tokens: {token_count} | Time: {time_seconds:.2f}s | ETA: {eta_str}",
            flush=True,
        )

        if args.verbose:
            print(f"  Expected class: {class_name}")
            print(f"  Output: {output_text}")

    total_time = time.time() - eval_start_time
    print("\n" + "=" * 60)
    print("SMILES RESULTS")
    print("=" * 60)
    print(f"Method: {'Unconstrained Baseline' if args.unconstrained else 'CSD'}")
    print(f"Model: {args.model}")
    print(f"Total Time: {total_time:.2f}s")
    print()
    print(metrics.summary())


if __name__ == "__main__":
    main()
