import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/aadivyar/csd-generation")

from synthesis.evaluate.evaluator import Evaluator
from synthesis.generate.generator import StrategyGenerator
from synthesis.run_synthesis import default_dafny_path
from synthesis.verify.compiler import DafnyCompiler
from synthesis.verify.verifier import DafnyVerifier


def main() -> None:
    repo = Path("/home/aadivyar/csd-generation")
    strategy_path = repo / "saved-results" / "2026-06-29-h29-gsm2b-six-variant-consensus-probe-body.dfy"
    if not strategy_path.exists():
        raise SystemExit(f"missing strategy path: {strategy_path}")

    out_root = repo / "outputs" / "generated" / "h29_gsm2b_six_variant_consensus_probe_20260629"
    python_dir = out_root / "python"
    results_dir = out_root / "results"
    dafny_dir = out_root / "dafny"
    for directory in (python_dir, results_dir, dafny_dir):
        directory.mkdir(parents=True, exist_ok=True)

    for name in (
        "AWS_BEARER_TOKEN_BEDROCK",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_PROFILE",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
    ):
        os.environ.pop(name, None)

    strategy = strategy_path.read_text()
    print("H29 strategy_chars", len(strategy), flush=True)
    print("H29 output_root", out_root, flush=True)

    generator = StrategyGenerator(model_name="local-template-only", backend="huggingface", device="cpu")
    full_code = generator.inject_strategy(strategy)
    (dafny_dir / "GeneratedCSD.dfy").write_text(full_code)
    (dafny_dir / "h29_gsm2b_six_variant_consensus_probe_20260629.dfy").write_text(full_code)
    print("H29 injected_full_code_chars", len(full_code), flush=True)

    verifier = DafnyVerifier(
        dafny_path=default_dafny_path(),
        timeout=180,
        extra_args=["--verification-time-limit", "120"],
    )
    compiler = DafnyCompiler(
        dafny_path=default_dafny_path(),
        output_dir=python_dir,
        timeout=180,
    )

    started = time.time()
    print("H29 verify_start", flush=True)
    verification = verifier.verify(full_code)
    print("H29 verify_success", verification.success, flush=True)
    if not verification.success:
        error = verification.get_error_summary()
        print(error[:2000], flush=True)
        (results_dir / "direct_eval_failure.json").write_text(
            json.dumps({"stage": "verification", "success": False, "error": error}, indent=2) + "\n"
        )
        raise SystemExit(2)

    print("H29 compile_start", flush=True)
    compilation = compiler.compile(full_code, "h29_gsm2b_six_variant_consensus_probe_20260629")
    print("H29 compile_success", compilation.success, flush=True)
    print("H29 compiled_module_path", compilation.main_module_path, flush=True)
    if not compilation.success or compilation.main_module_path is None:
        error = compilation.get_error_summary()
        print(error[:2000], flush=True)
        (results_dir / "direct_eval_failure.json").write_text(
            json.dumps({"stage": "compilation", "success": False, "error": error}, indent=2) + "\n"
        )
        raise SystemExit(3)

    print("H29 eval_start", flush=True)
    evaluator = Evaluator(
        dataset_name="gsm_symbolic",
        model_name="Qwen/Qwen3.5-2B",
        backend="vllm",
        device="cuda",
        sample_size=49,
        max_steps=900,
        step_token_budget=1,
        vllm_gpu_memory_utilization=0.25,
        vllm_tensor_parallel_size=1,
        gsm_split_file=str(repo / "environment" / "benchmark_splits" / "gsm_symbolic_crane_proportional_49x49_seed123.json"),
        gsm_split_name="train",
        max_seconds_per_example=600,
    )
    result = evaluator.evaluate_sample(
        compilation.main_module_path,
        sample_size=49,
        min_accuracy=0.0,
        early_stop_min_accuracy=0.0,
        early_stop_min_syntax_rate=0.0,
        min_examples_before_threshold_stop=49,
    )
    print(
        "H29 eval_done accuracy",
        result.accuracy,
        "syntax",
        result.syntax_rate,
        "num_examples",
        result.num_examples,
        flush=True,
    )
    payload = {
        "stage": "evaluation",
        "success": True,
        "accuracy": result.accuracy,
        "syntax_rate": result.syntax_rate,
        "num_examples": result.num_examples,
        "elapsed_seconds": time.time() - started,
        "compiled_module_path": str(compilation.main_module_path),
        "sample_outputs": result.sample_outputs,
    }
    (results_dir / "direct_eval_success.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("H29 saved", results_dir / "direct_eval_success.json", flush=True)


if __name__ == "__main__":
    main()
