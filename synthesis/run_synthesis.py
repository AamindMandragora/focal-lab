#!/usr/bin/env python3
"""
CLI entry point for CSD synthesis pipeline with evaluation feedback loop.

The pipeline runs: generate → verify → compile → runtime → evaluate → refine
until evaluation thresholds are met or max iterations exhausted.

Usage:
    python run_synthesis.py --task "..." --dataset gsm_symbolic \\
        --min-accuracy 0.3 --min-syntax-rate 0.5
"""

import argparse
import os
import sys
from pathlib import Path
try:
    from synthesis.project_defaults import default_dafny_path
except ImportError:
    from project_defaults import default_dafny_path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass


def main():
    parser = argparse.ArgumentParser(
        description="Synthesize constrained decoding strategies (default generation: OpenAI; Bedrock optional for Claude; eval often vLLM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # GSM-Symbolic
  python run_synthesis.py --task "Generate math reasoning strategy" \\
      --dataset gsm_symbolic \\
      --min-accuracy 0.3 --min-syntax-rate 0.5

  # With more iterations and custom eval sample size
  python run_synthesis.py --task "..." --dataset gsm_symbolic \\
      --min-accuracy 0.3 --min-syntax-rate 0.5 \\
      --output-name my_strategy --max-iterations 10 --eval-sample-size 20
"""
    )

    parser.add_argument(
        "--task", "-t",
        type=str,
        required=True,
        help="Task description for strategy generation"
    )

    parser.add_argument(
        "--max-iterations", "-n",
        type=int,
        default=5,
        help="Maximum refinement iterations (default: 5)"
    )

    parser.add_argument(
        "--generation-model",
        type=str,
        default=None,
        help="Model identifier for CSD generation (OpenAI model id when using --generation-backend openai; "
        "Bedrock model id when using bedrock; HF id for huggingface/vllm). "
        "OpenAI defaults from OPENAI_GENERATION_MODEL or gpt-5.4; Bedrock from BEDROCK_GENERATION_MODEL / AWS_BEDROCK_GENERATION_MODEL.",
    )

    parser.add_argument(
        "--generation-backend",
        type=str,
        choices=["huggingface", "vllm", "openai", "bedrock"],
        default="openai",
        help="Backend for strategy generation (default: openai)",
    )

    parser.add_argument(
        "--generation-api-base-url",
        type=str,
        default=None,
        help="Optional base URL for an API generation backend"
    )

    parser.add_argument(
        "--generation-api-key",
        type=str,
        default=None,
        help="Optional API key for generation. Defaults to the selected backend's environment variable."
    )

    parser.add_argument(
        "--eval-model",
        type=str,
        default="Qwen/Qwen2.5-Coder-7B-Instruct",
        help="Model for evaluation data generation/runtime (default: Qwen/Qwen2.5-Coder-7B-Instruct)"
    )

    parser.add_argument(
        "--eval-backend",
        type=str,
        choices=["huggingface", "vllm", "openai"],
        default="vllm",
        help="Backend for evaluation runtime (default: vllm; openai is unsupported for constrained runtime)."
    )

    parser.add_argument(
        "--output-name", "-o",
        type=str,
        default="generated_csd",
        help="Name for the output module (default: generated_csd)"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Base output directory (default: outputs/generated/). Each run writes into a unique subfolder."
    )

    parser.add_argument(
        "--baseline-output-dir",
        type=Path,
        default=None,
        help="Directory for baseline benchmark summaries (default: outputs/baselines/)"
    )

    parser.add_argument(
        "--initial-strategy-file",
        type=Path,
        default=None,
        help="Optional strategy body to use as the first attempt instead of asking the generation model for a fresh initial strategy.",
    )

    parser.add_argument(
        "--initial-attempt-offset",
        type=int,
        default=0,
        help="Attempt number offset for recovery runs seeded from an earlier synthesis attempt.",
    )

    parser.add_argument(
        "--dafny-path",
        type=str,
        default=default_dafny_path(),
        help="Path to Dafny executable"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for Qwen (default: 0.7)"
    )

    parser.add_argument(
        "--synthesis-max-tokens",
        "--max-tokens",
        dest="synthesis_max_tokens",
        type=int,
        default=2048,
        help="Maximum tokens for CSD synthesis generation per attempt (default: 2048)"
    )

    parser.add_argument(
        "--no-save-reports",
        action="store_true",
        help="Don't save failure/success reports to disk"
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "mps", "cpu", "auto"],
        default="auto",
        help="Device for model inference (default: auto)"
    )

    # Evaluation arguments (required - evaluation is part of the synthesis loop)
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        choices=["gsm_symbolic", "spider", "smiles"],
        required=True,
        help="Dataset to use for evaluation feedback (required)"
    )

    parser.add_argument(
        "--min-accuracy",
        type=float,
        required=True,
        help="Minimum accuracy threshold for evaluation (e.g. 0.3)"
    )

    parser.add_argument(
        "--min-syntax-rate",
        type=float,
        required=True,
        help="Minimum syntax validity rate threshold (e.g. 0.5)"
    )

    parser.add_argument(
        "--require-delimiters",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require evaluated outputs to contain at least one << >> span (default: true)"
    )

    parser.add_argument(
        "--eval-sample-size",
        type=int,
        default=10,
        help="Number of examples to evaluate on per iteration (default: 10)"
    )

    parser.add_argument(
        "--eval-seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducible evaluation sampling"
    )

    parser.add_argument(
        "--eval-max-steps",
        type=int,
        default=150,
        help="Maximum generation steps during evaluation (default: 150)"
    )

    parser.add_argument(
        "--eval-step-token-budget",
        type=int,
        default=1,
        help="Max tokens per outer generation step (1=token-level default, >1=symbol-level for structured outputs like SQL)"
    )

    parser.add_argument(
        "--eval-max-seconds-per-example",
        type=float,
        default=None,
        help="Optional runtime budget per evaluated example in seconds"
    )

    parser.add_argument(
        "--gsm-source-dir",
        type=str,
        default=None,
        help="Load GSM-Symbolic examples from this local folder of JSON files (e.g. CRANE's gsm_symbolic/) instead of HuggingFace"
    )

    parser.add_argument(
        "--gsm-split-file",
        type=str,
        default=None,
        help="Optional GSM train/eval split manifest with train_indices and eval_indices"
    )

    parser.add_argument(
        "--gsm-split-name",
        type=str,
        choices=["train", "eval"],
        default="train",
        help="Which split from --gsm-split-file to use during synthesis evaluation (default: train)"
    )

    parser.add_argument(
        "--spider-split-file",
        type=str,
        default=None,
        help="Optional Spider train/test split manifest with train_indices and test_indices"
    )

    parser.add_argument(
        "--spider-split-name",
        type=str,
        choices=["train", "test", "eval"],
        default="train",
        help="Which split from --spider-split-file to use during synthesis evaluation (default: train)"
    )

    parser.add_argument(
        "--grammars-dir",
        type=Path,
        default=None,
        help="Optional override for built-in grammar directory (default: synthesis/evaluate/grammars or CSD_GRAMMARS_DIR)"
    )

    parser.add_argument(
        "--smiles-samples-per-class",
        type=int,
        default=10,
        help="Number of SMILES attempts per molecular class during synthesis feedback (default: 10)"
    )

    parser.add_argument(
        "--smiles-final-samples-per-class",
        type=int,
        default=100,
        help="Target valid unique SMILES samples per class for final benchmark scripts (default: 100)"
    )

    parser.add_argument(
        "--smiles-classes",
        type=str,
        default="acrylates,chain_extenders,isocyanates",
        help="Comma-separated SMILES classes to evaluate (default: all three CARS molecule classes)"
    )

    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load generation model in 4-bit quantization"
    )

    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Load generation model in 8-bit quantization"
    )

    parser.add_argument(
        "--vllm-tensor-parallel-size",
        type=int,
        default=None,
        help="Explicit tensor parallel size for vLLM. Defaults to visible GPU count."
    )

    parser.add_argument(
        "--vllm-pipeline-parallel-size",
        type=int,
        default=1,
        help="Explicit pipeline parallel size for vLLM (default: 1)"
    )

    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=0.8,
        help="GPU memory fraction reserved by vLLM (default: 0.8)"
    )

    parser.add_argument(
        "--vllm-max-model-len",
        type=int,
        default=4096,
        help="Maximum model context length passed to vLLM (default: 4096)"
    )

    parser.add_argument(
        "--vllm-enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable torch.compile and CUDA graphs in vLLM for stability (default: true)"
    )

    parser.add_argument(
        "--adaptive-helper-mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable empirical helper-call masking/pruning in synthesis prompts (default: true)"
    )

    parser.add_argument(
        "--helper-selection-policy",
        type=str,
        choices=["utility", "bandit"],
        default="utility",
        help="Helper selection policy for adaptive masking (default: utility)"
    )

    parser.add_argument(
        "--helper-mask-min-evals",
        type=int,
        default=4,
        help="Minimum evaluated attempts before helper pruning activates (default: 4)"
    )

    parser.add_argument(
        "--helper-mask-min-uses",
        type=int,
        default=2,
        help="Minimum helper usage count before that helper can be pruned (default: 2)"
    )

    parser.add_argument(
        "--helper-mask-margin",
        type=float,
        default=0.25,
        help="Prune helpers whose mean utility is below run mean by this margin (default: 0.25)"
    )

    parser.add_argument(
        "--helper-mask-max-disabled",
        type=int,
        default=6,
        help="Maximum helpers disabled by empirical pruning in a run (default: 6)"
    )

    parser.add_argument(
        "--helper-bandit-min-evals",
        type=int,
        default=3,
        help="Minimum evaluated attempts before bandit helper selection activates (default: 3)"
    )

    parser.add_argument(
        "--helper-bandit-top-k",
        type=int,
        default=12,
        help="Number of prunable helpers kept active under bandit selection (default: 12)"
    )

    parser.add_argument(
        "--helper-bandit-ucb-c",
        type=float,
        default=0.35,
        help="UCB exploration coefficient for bandit helper selection (default: 0.35)"
    )

    parser.add_argument(
        "--helper-bandit-explore-untried",
        type=int,
        default=1,
        help="Number of unseen helpers to force-explore per selection step (default: 1)"
    )

    parser.add_argument(
        "--refinement-beam-size",
        type=int,
        default=1,
        help="Number of refinement candidates sampled per failure (default: 1)"
    )

    parser.add_argument(
        "--local-neighborhood-refinement",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer local strategy edits when selecting among beam candidates (default: true)"
    )

    parser.add_argument(
        "--max-local-edit-ratio",
        type=float,
        default=0.65,
        help="Soft local-edit bound for beam selection as changed-lines ratio (default: 0.65)"
    )

    parser.add_argument(
        "--beam-verify-candidates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Verify beam candidates before selecting one (default: true)"
    )

    args = parser.parse_args()

    if args.generation_model is None:
        if args.generation_backend == "bedrock":
            resolved = os.environ.get("BEDROCK_GENERATION_MODEL") or os.environ.get(
                "AWS_BEDROCK_GENERATION_MODEL"
            )
            if not resolved:
                parser.error(
                    "Bedrock synthesis requires --generation-model or "
                    "BEDROCK_GENERATION_MODEL / AWS_BEDROCK_GENERATION_MODEL in the environment."
                )
            args.generation_model = resolved
        elif args.generation_backend == "openai":
            args.generation_model = os.environ.get("OPENAI_GENERATION_MODEL") or "gpt-5.4"
        else:
            from synthesis.generate.generator import StrategyGenerator as _StrategyGenerator

            args.generation_model = _StrategyGenerator.DEFAULT_MODEL

    if args.generation_backend == "vllm" or args.eval_backend == "vllm":
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    # Normalize output_dir if provided (handle potential backslashes from user input)
    if args.output_dir:
        args.output_dir = Path(str(args.output_dir).replace("\\", "/"))
    else:
        args.output_dir = Path(
            os.environ.get(
                "CSD_OUTPUT_DIR",
                str(Path(__file__).resolve().parent.parent / "outputs" / "generated"),
            )
        )
    if args.baseline_output_dir:
        args.baseline_output_dir = Path(str(args.baseline_output_dir).replace("\\", "/"))
    else:
        args.baseline_output_dir = Path(
            os.environ.get(
                "CSD_BASELINE_OUTPUT_DIR",
                str(Path(__file__).resolve().parent.parent / "outputs" / "baselines"),
            )
        )

    # Root output layout:
    # - outputs/generated/: synthesis run artifacts (dafny/python/results)
    # - outputs/baselines/: baseline benchmark summaries
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.baseline_output_dir.mkdir(parents=True, exist_ok=True)

    # Import here to avoid loading heavy dependencies if just showing help
    from synthesis.generate.generator import StrategyGenerator
    from synthesis.verify.verifier import DafnyVerifier
    from synthesis.verify.compiler import DafnyCompiler
    from synthesis.evaluate.evaluator import Evaluator
    from synthesis.evaluate.feedback_loop import SynthesisPipeline, SynthesisExhaustionError

    # Create components
    print("Initializing synthesis pipeline...")

    device = None if args.device == "auto" else args.device

    generator = StrategyGenerator(
        model_name=args.generation_model,
        backend=args.generation_backend,
        device=device,
        max_new_tokens=args.synthesis_max_tokens,
        temperature=args.temperature,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        vllm_pipeline_parallel_size=args.vllm_pipeline_parallel_size,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_max_model_len=args.vllm_max_model_len,
        vllm_enforce_eager=args.vllm_enforce_eager,
        api_base_url=args.generation_api_base_url,
        api_key=args.generation_api_key,
    )

    verifier = DafnyVerifier(
        dafny_path=args.dafny_path,
        timeout=180,
        extra_args=["--verification-time-limit", "120"],
    )
    # Compiler output dir is set per-run inside the pipeline (so runs don't overwrite each other).
    compiler = DafnyCompiler(dafny_path=args.dafny_path, output_dir=args.output_dir)
    # Runner is created by the pipeline with task-appropriate parser mode

    feedback_sample_size = (
        args.smiles_samples_per_class if args.dataset == "smiles" else args.eval_sample_size
    )

    # Create evaluator for the feedback loop
    print(f"Setting up evaluator for dataset: {args.dataset}")
    print(f"  Generation model: {args.generation_model}")
    print(f"  Evaluation model: {args.eval_model}")
    evaluator = Evaluator(
        dataset_name=args.dataset,
        model_name=args.eval_model,
        backend=args.eval_backend,
        device=device or "cuda",
        sample_size=feedback_sample_size,
        max_steps=args.eval_max_steps,
        step_token_budget=args.eval_step_token_budget,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        vllm_pipeline_parallel_size=args.vllm_pipeline_parallel_size,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_max_model_len=args.vllm_max_model_len,
        vllm_enforce_eager=args.vllm_enforce_eager,
        sample_seed=args.eval_seed,
        max_seconds_per_example=args.eval_max_seconds_per_example,
        gsm_source_dir=args.gsm_source_dir,
        gsm_split_file=args.gsm_split_file,
        gsm_split_name=args.gsm_split_name,
        spider_split_file=args.spider_split_file,
        spider_split_name=args.spider_split_name,
        smiles_classes=args.smiles_classes,
        grammars_dir=args.grammars_dir,
    )

    pipeline = SynthesisPipeline(
        evaluator=evaluator,
        generator=generator,
        verifier=verifier,
        compiler=compiler,
        runner=None,  # Let pipeline create task-appropriate runner
        max_iterations=args.max_iterations,
        output_dir=args.output_dir,
        save_reports=not args.no_save_reports,
        # Evaluation thresholds
        min_accuracy=args.min_accuracy,
        min_syntax_rate=args.min_syntax_rate,
        require_delimiters=False if args.dataset == "smiles" else args.require_delimiters,
        eval_sample_size=feedback_sample_size,
        eval_max_seconds_per_example=args.eval_max_seconds_per_example,
        adaptive_helper_mask=args.adaptive_helper_mask,
        helper_selection_policy=args.helper_selection_policy,
        helper_mask_min_evals=args.helper_mask_min_evals,
        helper_mask_min_uses=args.helper_mask_min_uses,
        helper_mask_margin=args.helper_mask_margin,
        helper_mask_max_disabled=args.helper_mask_max_disabled,
        helper_bandit_min_evals=args.helper_bandit_min_evals,
        helper_bandit_top_k=args.helper_bandit_top_k,
        helper_bandit_ucb_c=args.helper_bandit_ucb_c,
        helper_bandit_explore_untried=args.helper_bandit_explore_untried,
        refinement_beam_size=args.refinement_beam_size,
        local_neighborhood_refinement=args.local_neighborhood_refinement,
        max_local_edit_ratio=args.max_local_edit_ratio,
        beam_verify_candidates=args.beam_verify_candidates,
    )

    initial_strategy_code = None
    if args.initial_strategy_file:
        initial_strategy_code = args.initial_strategy_file.read_text()
        print(f"Loaded initial strategy seed from: {args.initial_strategy_file}")

    # Run synthesis
    try:
        result = pipeline.synthesize(
            task_description=args.task,
            output_name=args.output_name,
            initial_strategy_code=initial_strategy_code,
            initial_attempt_offset=args.initial_attempt_offset,
        )

        print("\n" + "=" * 60)
        print("SYNTHESIS COMPLETE")
        print("=" * 60)
        print(f"Strategy: {result.strategy_code}")
        print(f"Compiled module: {result.compiled_module_path}")
        print(f"Output directory: {result.output_dir}")
        if getattr(result, "run_dir", None):
            print(f"Run directory: {result.run_dir}")
        print(f"Total attempts: {len(result.attempts)}")
        print(f"Total time: {result.total_time_ms:.1f}ms")

        sys.exit(0)

    except SynthesisExhaustionError as e:
        print("\n" + "=" * 60)
        print("SYNTHESIS FAILED")
        print("=" * 60)
        print(e.get_failure_summary())

        sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nSynthesis interrupted by user")
        sys.exit(130)

    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
