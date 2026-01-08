#!/usr/bin/env python3
"""
CLI entry point for CSD synthesis pipeline with reusable LM for batch dataset runs.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import secrets
import json
import time
import random
import torch

# ------------------------------
# Import CSDRunner from your refactored run_csd_with_grammar
# ------------------------------
from scripts.run_csd_with_grammar import CSDRunner

def main():
    parser = argparse.ArgumentParser(
        description="Synthesize or run constrained decoding strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ======================
    # Synthesis arguments
    # ======================
    parser.add_argument("--task", "-t", type=str, help="Task description for strategy generation")
    parser.add_argument("--max-iterations", "-n", type=int, default=5)
    parser.add_argument("--model", "-m", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--output-name", "-o", type=str, default="generated_csd")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dafny-path", type=str, default="dafny")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--no-save-reports", action="store_true")
    parser.add_argument("--device", choices=["cuda", "mps", "cpu", "auto"], default="auto")

    # ======================
    # Mode switches
    # ======================
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--compile-only", action="store_true")

    # ======================
    # Dataset execution
    # ======================
    parser.add_argument("--dataset", type=Path, help="JSONL dataset for batch execution")
    parser.add_argument("--compiled-module", type=Path, help="Compiled Dafny Python module to execute")
    parser.add_argument("--dataset-output", type=Path, help="Where to write JSONL results")
    parser.add_argument("--parser-mode", choices=["permissive", "json", "math"], default="permissive",
                        help="Parser mode for dataset execution")
    parser.add_argument("--lark-file", type=Path, default=None, help="Path to Lark grammar file (required for math mode)")

    args = parser.parse_args()

    # ======================
    # Dataset mode (early exit)
    # ======================
    if args.dataset is not None:
        run_dataset_mode(args)
        return

    # ======================
    # Verify / compile only
    # ======================
    if args.verify_only or args.compile_only:
        run_verification_only(args)
        return

    # ======================
    # Normal synthesis
    # ======================
    if not args.task:
        print("Error: --task is required unless running dataset mode")
        sys.exit(1)

    from synthesis.generator import StrategyGenerator
    from synthesis.verifier import DafnyVerifier
    from synthesis.compiler import DafnyCompiler
    from synthesis.runner import StrategyRunner
    from synthesis.feedback_loop import SynthesisPipeline, SynthesisExhaustionError

    device = None if args.device == "auto" else args.device

    generator = StrategyGenerator(
        model_name=args.model,
        device=device,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    verifier = DafnyVerifier(dafny_path=args.dafny_path)
    compiler = DafnyCompiler(dafny_path=args.dafny_path, output_dir=args.output_dir)
    runner = StrategyRunner()

    pipeline = SynthesisPipeline(
        generator=generator,
        verifier=verifier,
        compiler=compiler,
        runner=runner,
        max_iterations=args.max_iterations,
        output_dir=args.output_dir,
        save_reports=not args.no_save_reports,
    )

    try:
        result = pipeline.synthesize(
            task_description=args.task,
            output_name=args.output_name,
        )

        print("\nSYNTHESIS COMPLETE")
        print(f"Compiled module: {result.compiled_module_path}")
        print(f"Total attempts: {len(result.attempts)}")
        print(f"Total time: {result.total_time_ms:.1f}ms")
        sys.exit(0)

    except SynthesisExhaustionError as e:
        print("SYNTHESIS FAILED")
        print(e.get_failure_summary())
        sys.exit(1)


# ============================================================
# Dataset execution using CSDRunner
# ============================================================

def run_dataset_mode(args):
    import json
    import time

    if args.compiled_module is None:
        print("Error: --compiled-module is required for dataset runs")
        sys.exit(1)

    if not args.dataset.exists():
        print(f"Dataset not found: {args.dataset}")
        sys.exit(1)

    output_path = args.dataset_output or (
        args.compiled_module.parent / "dataset_results.jsonl"
    )

    # ------------------------------
    # Initialize runner once
    # ------------------------------
    device = args.device

    runner = CSDRunner.get_instance()
    grammar_source = args.lark_file if args.lark_file else args.parser_mode
    runner.initialize(
        run_dir=args.compiled_module.parent,
        grammar_source=grammar_source,
        lm_name=args.model,
        device=device
    )

    # ------------------------------
    # Run dataset
    # ------------------------------
    total = 0
    successes = 0
    total_time = 0.0

    with args.dataset.open() as f, output_path.open("w") as out:
        for line in f:
            item = json.loads(line)
            prompt = item.get("question", "")
            example_id = item.get("id")

            start = time.time()
            result = runner.run_prompt(prompt=prompt, max_steps=args.max_tokens)
            elapsed = (time.time() - start) * 1000

            record = {
                "id": example_id,
                "success": result.get("success", False),
                "output": result.get("output_text", ""),
                "error_message": result.get("error", None),
                "execution_time_ms": elapsed,
            }

            out.write(json.dumps(record) + "\n")

            total += 1
            total_time += elapsed
            if result.get("success"):
                successes += 1

    print("\nDATASET RUN COMPLETE")
    print(f"Parser mode: {args.parser_mode}")
    if args.lark_file:
        print(f"Lark grammar: {args.lark_file}")
    print(f"Total examples: {total}")
    print(f"Success rate: {successes / max(total,1):.3f}")
    print(f"Avg time (ms): {total_time / max(total,1):.1f}")
    print(f"Results written to: {output_path}")


# ============================================================
# Verification / compilation only
# ============================================================

def run_verification_only(args):
    from synthesis.verifier import DafnyVerifier
    from synthesis.compiler import DafnyCompiler
    from synthesis.runner import StrategyRunner

    dafny_file = Path(__file__).parent / "dafny" / "GeneratedCSD.dfy"

    if not dafny_file.exists():
        print(f"Error: {dafny_file} not found")
        sys.exit(1)

    verifier = DafnyVerifier(dafny_path=args.dafny_path)
    result = verifier.verify_file(dafny_file)

    if not result.success:
        print("Verification failed")
        print(result.get_error_summary())
        sys.exit(1)

    if args.verify_only:
        sys.exit(0)

    base_output_dir = args.output_dir or (Path(__file__).parent / "outputs" / "generated-csd")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(3)
    run_dir = base_output_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    compiler = DafnyCompiler(dafny_path=args.dafny_path, output_dir=run_dir)
    compile_result = compiler.compile_file(dafny_file, args.output_name)

    if not compile_result.success:
        print("Compilation failed")
        print(compile_result.get_error_summary())
        sys.exit(1)

    if compile_result.main_module_path:
        runner = StrategyRunner()
        runtime_result = runner.run(compile_result.main_module_path)

        if not runtime_result.success:
            print("Runtime error")
            print(runtime_result.get_error_summary())
            sys.exit(1)

    print("Verification + compilation successful")
    sys.exit(0)


if __name__ == "__main__":
    main()