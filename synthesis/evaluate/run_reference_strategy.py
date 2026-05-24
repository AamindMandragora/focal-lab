"""
Evaluate an already-compiled GeneratedCSD module on any benchmark.

Reference ``.dfy`` files under ``synthesis/verify/reference/`` are documentation
only — this entrypoint never compiles them. Pass a ``GeneratedCSD.py`` from a
synthesis run (or another verified build) via ``--compiled-module``.

Usage:
  python -m synthesis.evaluate.run_reference_strategy \\
    --compiled-module outputs/synthesis_runs/<run>/generated_csd/GeneratedCSD.py \\
    --strategy crane --dataset smiles \\
    --eval-model Qwen/Qwen2.5-Coder-7B-Instruct \\
    --eval-sample-size 50 --eval-max-steps 900 \\
    --output-json outputs/baselines/crane/smiles.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from synthesis.evaluate.baseline_store import build_minimal_baseline_record
from synthesis.evaluate.evaluator import Evaluator


STRATEGY_NAMES = frozenset(
    {"unconstrained", "gcd", "crane", "itergen", "cars", "rejection_sampling"}
)


def _evaluate(
    compiled_module: Path,
    dataset: str,
    eval_model: str,
    eval_backend: str,
    device: str,
    sample_size: int,
    max_steps: int,
    step_token_budget: int,
    vllm_gpu_memory_utilization: float,
    vllm_max_model_len: int | None,
    smiles_classes: str | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = dict(
        dataset_name=dataset,
        model_name=eval_model,
        backend=eval_backend,
        device=device,
        sample_size=sample_size,
        max_steps=max_steps,
        step_token_budget=step_token_budget,
        vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
        prompt_tier=1,
    )
    if vllm_max_model_len is not None:
        kwargs["vllm_max_model_len"] = vllm_max_model_len
    if smiles_classes is not None:
        kwargs["smiles_classes"] = smiles_classes

    evaluator = Evaluator(**kwargs)
    try:
        result = evaluator.evaluate_sample(compiled_module, sample_size=sample_size)
    finally:
        evaluator.unload_runtime()

    payload = build_minimal_baseline_record(result, dataset=dataset)
    payload.update(
        {
            "num_correct": result.num_correct,
            "num_examples": result.num_examples,
            "contains_delimiters": result.contains_delimiters,
        }
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compiled-module",
        type=Path,
        required=True,
        help="Path to an existing GeneratedCSD.py (synthesis output; references are not compiled here)",
    )
    parser.add_argument(
        "--strategy",
        required=True,
        choices=sorted(STRATEGY_NAMES),
        help="Label for the run (metadata only; module must already match)",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["gsm_symbolic", "spider", "smiles"],
    )
    parser.add_argument("--eval-model", required=True)
    parser.add_argument("--eval-backend", default="vllm")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-sample-size", type=int, default=50)
    parser.add_argument("--eval-max-steps", type=int, default=900)
    parser.add_argument("--eval-step-token-budget", type=int, default=1)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--vllm-max-model-len", type=int, default=None)
    parser.add_argument("--smiles-classes", default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    compiled_path = args.compiled_module.resolve()
    if not compiled_path.is_file():
        raise SystemExit(
            f"Compiled module not found: {compiled_path}\n"
            "Reference .dfy strategies are never compiled by this tool; "
            "use a GeneratedCSD.py from synthesis or another existing build."
        )

    print(
        f"Evaluating {args.strategy} on {args.dataset} with {args.eval_model} "
        f"(module={compiled_path}, n={args.eval_sample_size}, steps={args.eval_max_steps})",
        flush=True,
    )
    payload = _evaluate(
        compiled_module=compiled_path,
        dataset=args.dataset,
        eval_model=args.eval_model,
        eval_backend=args.eval_backend,
        device=args.device,
        sample_size=args.eval_sample_size,
        max_steps=args.eval_max_steps,
        step_token_budget=args.eval_step_token_budget,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_max_model_len=args.vllm_max_model_len,
        smiles_classes=args.smiles_classes,
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"Wrote {args.output_json}: accuracy={payload['accuracy']:.3f} "
        f"syntax_rate={payload['syntax_rate']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
