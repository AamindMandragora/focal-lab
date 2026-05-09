"""
Compile a verified reference Dafny strategy and evaluate it on any benchmark.

This replaces legacy external codebases with our own verified Dafny
reference implementations, ensuring consistent evaluation infrastructure
across all strategy-benchmark combinations.

Available strategies:
  gcd       Pure hard-mask constrained decoding (SynCode-style)
  crane     Adaptive constrained-unconstrained switching (CRANE-style)
  itergen   Chunked constrained symbol generation (IterGen-style)
  cars      Adaptive group-boosted constrained steps (CARS-style)

Usage:
  python -m synthesis.evaluate.run_reference_strategy \\
    --strategy crane --dataset smiles \\
    --eval-model Qwen/Qwen2.5-Coder-7B-Instruct \\
    --eval-sample-size 50 --eval-max-steps 900 \\
    --output-json outputs/baselines/crane/smiles.json
"""
from __future__ import annotations

import argparse
import json
import re
import tempfile
from pathlib import Path
from typing import Any


STRATEGY_DFY: dict[str, str] = {
    "gcd": "gcd.dfy",
    "crane": "crane.dfy",
    "itergen": "itergen.dfy",
    "cars": "cars.dfy",
}

REFERENCE_DIR = Path(__file__).resolve().parents[1] / "verify" / "reference"


def _rewrite_to_generated_csd(source_text: str) -> str:
    """Rewrite the module name so the evaluator can import it as GeneratedCSD."""
    return re.sub(
        r"module\s+Reference\w+CSD\s*\{",
        "module GeneratedCSD {",
        source_text,
        count=1,
    )


def _compile_reference(strategy: str, output_dir: Path) -> Path:
    """Compile a reference .dfy to Python under output_dir, return GeneratedCSD.py path."""
    from synthesis.verify.compiler import DafnyCompiler

    dfy_name = STRATEGY_DFY[strategy]
    source_path = REFERENCE_DIR / dfy_name
    if not source_path.is_file():
        raise FileNotFoundError(f"Reference strategy not found: {source_path}")

    source_text = source_path.read_text()
    source_text = _rewrite_to_generated_csd(source_text)

    compiler = DafnyCompiler(output_dir=output_dir)
    result = compiler.compile(source_text, output_name=f"ref_{strategy}")
    if not result.success:
        errors = "; ".join(e.message for e in result.errors[:5])
        raise RuntimeError(f"Failed to compile {dfy_name}: {errors}")

    if result.main_module_path is None:
        raise RuntimeError(f"Compilation produced no main module for {dfy_name}")

    return result.main_module_path


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
    from synthesis.evaluate.evaluator import Evaluator

    kwargs: dict[str, Any] = dict(
        dataset_name=dataset,
        model_name=eval_model,
        backend=eval_backend,
        device=device,
        sample_size=sample_size,
        max_steps=max_steps,
        step_token_budget=step_token_budget,
        vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
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

    answers: list[dict[str, str]] = []
    for sample in result.sample_outputs or []:
        question = str(sample.get("question", ""))
        generated = sample.get("actual")
        if generated is None:
            generated = sample.get("full_output", "")
        answers.append({"question": question, "generated_answer": str(generated)})

    return {
        "accuracy": float(result.accuracy),
        "syntax_rate": float(result.syntax_rate),
        "num_correct": result.num_correct,
        "num_examples": result.num_examples,
        "contains_delimiters": result.contains_delimiters,
        "answers": answers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strategy",
        required=True,
        choices=sorted(STRATEGY_DFY),
        help="Reference strategy to evaluate",
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
    parser.add_argument(
        "--compiled-cache-dir",
        type=Path,
        default=None,
        help="Directory to cache compiled reference modules (avoids recompilation)",
    )
    args = parser.parse_args()

    cache_dir = args.compiled_cache_dir
    if cache_dir is None:
        repo_root = Path(__file__).resolve().parents[2]
        cache_dir = repo_root / "outputs" / "compiled_references"
    cache_dir.mkdir(parents=True, exist_ok=True)

    strategy_cache = cache_dir / args.strategy
    generated_csd = strategy_cache / f"ref_{args.strategy}" / "GeneratedCSD.py"
    if not generated_csd.is_file():
        print(f"Compiling reference strategy: {args.strategy}", flush=True)
        compiled_path = _compile_reference(args.strategy, strategy_cache)
        print(f"  Compiled to: {compiled_path}", flush=True)
    else:
        compiled_path = generated_csd
        print(f"Using cached compilation: {compiled_path}", flush=True)

    print(
        f"Evaluating {args.strategy} on {args.dataset} with {args.eval_model} "
        f"(n={args.eval_sample_size}, steps={args.eval_max_steps})",
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
