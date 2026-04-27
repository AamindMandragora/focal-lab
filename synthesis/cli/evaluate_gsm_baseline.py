#!/usr/bin/env python3
"""Evaluate a fixed GSM baseline run.

The old built-in starter strategy has been removed from synthesis so it cannot
seed generated CSDs. This command now refuses to synthesize a canned baseline
and points callers at existing-run evaluation instead.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deprecated fixed GSM baseline evaluator.",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-Coder-7B-Instruct",
        help="Evaluation model name.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device passed to the GSM evaluator.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Number of examples to evaluate.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1024,
        help="Maximum generation steps during evaluation.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=3000,
        help="Vocabulary size for evaluation.",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load the evaluation model in 4-bit mode.",
    )
    parser.add_argument(
        "--debug-delimiters",
        action="store_true",
        help="Enable delimiter debugging in the evaluator.",
    )
    return parser


def _build_baseline_run() -> Path:
    raise RuntimeError(
        "The fixed starter baseline was removed so synthesis no longer seeds the model "
        "with canned strategy code. Re-evaluate an existing run with "
        "`python -m synthesis.cli.evaluate_existing_run --run-dir <run-dir>` instead."
    )


def main() -> int:
    args = _build_arg_parser().parse_args()
    run_dir = _build_baseline_run()

    command = [
        sys.executable,
        "-m",
        "evaluation.gsm_symbolic.cli",
        "--run-dir",
        str(run_dir),
        "--model",
        args.model,
        "--device",
        args.device,
        "--limit",
        str(args.limit),
        "--max-steps",
        str(args.max_steps),
        "--vocab-size",
        str(args.vocab_size),
    ]
    if args.load_in_4bit:
        command.append("--load-in-4bit")
    if args.debug_delimiters:
        command.append("--debug-delimiters")

    print(f"Built baseline run in {run_dir}")
    result = subprocess.run(command, cwd=PROJECT_ROOT, env=os.environ.copy())
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
