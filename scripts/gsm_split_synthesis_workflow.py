#!/usr/bin/env python3
"""Train/eval split workflow for GSM-Symbolic CSD synthesis.

This script keeps synthesis feedback and held-out reporting separate:

1. `prepare` creates a deterministic 50/50 split over the sorted CRANE GSM
   folder, runs the CRANE baseline on the training half, and writes a
   `run_synthesis.py` command whose thresholds require strictly improving over
   CRANE on that same training half.
2. `heldout` evaluates CRANE and a synthesized CSD on the held-out half.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from project_defaults import (
    default_cars_repo,
    default_crane_repo,
    default_dafny_path,
    default_gsm_source_dir,
    default_itergen_repo,
)

DEFAULT_CRANE_GSM_DIR = default_gsm_source_dir()

from gpu_utils import cuda_env, select_cuda_visible_devices, visible_device_count
from baseline_cache import (
    accuracy_syntax_validator,
    file_digest,
    reuse_cached_baseline,
    store_cached_baseline,
)

DEFAULT_TASK = (
    "Solve math word problems step by step, writing each arithmetic computation "
    "inside << >> delimiters."
)


def strict_next_threshold(pass_rate: float, n_examples: int) -> float:
    """Smallest discrete rate that is strictly greater than pass_rate."""
    if n_examples <= 0:
        raise ValueError("n_examples must be positive")
    passed = int(round(pass_rate * n_examples))
    return min(1.0, (passed + 1) / n_examples)


def parse_difficulty_counts(raw: str) -> dict[str, int]:
    """Parse easy/medium/hard counts from 'easy=17,medium=17,hard=16' or '17,17,16'."""
    raw = raw.strip()
    if not raw:
        raise ValueError("Difficulty counts cannot be empty")
    difficulties = ("easy", "medium", "hard")
    if "=" not in raw:
        parts = [part.strip() for part in raw.split(",") if part.strip()]
        if len(parts) != 3:
            raise ValueError(
                "Difficulty counts must be easy,medium,hard or easy=N,medium=N,hard=N"
            )
        return {difficulty: int(value) for difficulty, value in zip(difficulties, parts)}

    counts: dict[str, int] = {}
    for part in raw.split(","):
        if not part.strip():
            continue
        key, value = part.split("=", 1)
        key = key.strip().lower()
        if key not in difficulties:
            raise ValueError(f"Unknown GSM difficulty: {key}")
        counts[key] = int(value.strip())
    missing = [difficulty for difficulty in difficulties if difficulty not in counts]
    if missing:
        raise ValueError(f"Missing difficulty counts for: {missing}")
    return counts


def split_size(split_file: Path, split_name: str) -> int:
    manifest = json.loads(split_file.read_text())
    key = f"{split_name}_indices"
    if key not in manifest:
        raise ValueError(f"{split_file} does not contain {key}")
    return len(manifest[key])


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def evaluate_module(
    *,
    module_path: Path,
    label: str,
    split_file: Path,
    split_name: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    from synthesis.evaluator import Evaluator

    n_examples = split_size(split_file, split_name)
    evaluator = Evaluator(
        dataset_name="gsm_symbolic",
        model_name=args.eval_model,
        backend=args.eval_backend,
        device=args.device,
        sample_size=n_examples,
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
        gsm_split_file=split_file,
        gsm_split_name=split_name,
    )
    try:
        result = evaluator.evaluate_sample(module_path, sample_size=n_examples)
        return {
            "label": label,
            "module_path": str(module_path),
            "split_file": str(split_file),
            "split_name": split_name,
            "success": result.success,
            "accuracy": result.accuracy,
            "syntax_rate": result.syntax_rate,
            "contains_delimiters": result.contains_delimiters,
            "num_correct": result.num_correct,
            "num_examples": result.num_examples,
            "total_time_seconds": result.total_time_seconds,
            "max_sample_time_seconds": result.max_sample_time_seconds,
            "error": result.error,
            "sample_outputs": result.sample_outputs,
        }
    finally:
        evaluator.unload_runtime()


def crane_benchmark_command(
    args: argparse.Namespace,
    split_file: Path,
    split_name: str,
    output_path: Path,
) -> list[str]:
    return [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "benchmark_crane_baseline.py"),
        "--dataset",
        "gsm_symbolic",
        "--output",
        str(output_path),
        "--output-dir",
        str(args.output_dir),
        "--crane-repo",
        str(args.crane_repo),
        "--crane-device",
        args.crane_device,
        "--gsm-split-file",
        str(split_file),
        "--gsm-split-name",
        split_name,
        "--gsm-source-dir",
        str(args.gsm_source_dir),
        "--sample-size",
        str(split_size(split_file, split_name)),
        "--eval-model",
        args.eval_model,
        "--eval-backend",
        args.eval_backend,
        "--device",
        args.device,
        "--eval-max-steps",
        str(args.eval_max_steps),
        "--eval-step-token-budget",
        str(args.eval_step_token_budget),
        "--vllm-gpu-memory-utilization",
        str(args.vllm_gpu_memory_utilization),
        "--vllm-max-model-len",
        str(args.vllm_max_model_len),
    ]


def build_synthesis_command(
    *,
    split_file: Path,
    train_size: int,
    min_accuracy: float,
    min_syntax_rate: float,
    args: argparse.Namespace,
) -> list[str]:
    cmd = [
        sys.executable,
        "run_synthesis.py",
        "--task",
        args.task,
        "--dataset",
        "gsm_symbolic",
        "--gsm-source-dir",
        args.gsm_source_dir,
        "--gsm-split-file",
        str(split_file),
        "--gsm-split-name",
        "train",
        "--max-iterations",
        str(args.max_iterations),
        "--generation-model",
        args.generation_model,
        "--generation-backend",
        args.generation_backend,
        "--eval-model",
        args.eval_model,
        "--eval-backend",
        args.eval_backend,
        "--output-name",
        args.synthesis_output_name,
        "--min-accuracy",
        f"{min_accuracy:.12g}",
        "--min-syntax-rate",
        f"{min_syntax_rate:.12g}",
        "--eval-sample-size",
        str(train_size),
        "--eval-max-steps",
        str(args.eval_max_steps),
        "--eval-step-token-budget",
        str(args.eval_step_token_budget),
        "--device",
        args.device,
        "--dafny-path",
        args.dafny_path,
        "--synthesis-max-tokens",
        str(args.synthesis_max_tokens),
        "--vllm-gpu-memory-utilization",
        str(args.vllm_gpu_memory_utilization),
        "--vllm-max-model-len",
        str(args.vllm_max_model_len),
    ]
    if args.eval_seed is not None:
        cmd.extend(["--eval-seed", str(args.eval_seed)])
    if args.eval_max_seconds_per_example is not None:
        cmd.extend([
            "--eval-max-seconds-per-example",
            str(args.eval_max_seconds_per_example),
        ])
    if args.vllm_tensor_parallel_size is not None:
        cmd.extend(["--vllm-tensor-parallel-size", str(args.vllm_tensor_parallel_size)])
    if args.vllm_pipeline_parallel_size != 1:
        cmd.extend(["--vllm-pipeline-parallel-size", str(args.vllm_pipeline_parallel_size)])
    if args.vllm_enforce_eager:
        cmd.append("--vllm-enforce-eager")
    else:
        cmd.append("--no-vllm-enforce-eager")
    if args.load_in_4bit:
        cmd.append("--load-in-4bit")
    if args.load_in_8bit:
        cmd.append("--load-in-8bit")
    return cmd


def command_text(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def run_logged(cmd: list[str], log_path: Path, *, env: dict[str, str] | None = None) -> None:
    print(command_text(cmd))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env or os.environ.copy(),
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log_file.write(line)
            log_file.flush()
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"Command failed with exit code {rc}. See log: {log_path}")


def original_framework_env(args: argparse.Namespace, requested: str | None = None, *, count: int = 1) -> dict[str, str]:
    return cuda_env(
        requested=requested or args.original_framework_cuda_visible_devices,
        count=count,
        min_free_mib=args.gpu_min_free_mib,
        avoid=args.gpu_avoid_devices,
    )


def itergen_train_command(args: argparse.Namespace, split_file: Path, output_path: Path) -> list[str]:
    return [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_itergen_gsm_split.py"),
        "--itergen-repo",
        str(args.itergen_repo),
        "--split-file",
        str(split_file),
        "--split-name",
        "train",
        "--gsm-source-dir",
        str(args.gsm_source_dir),
        "--model",
        args.eval_model,
        "--device",
        args.itergen_device,
        "--seed",
        str(args.itergen_seed),
        "--recurrence-penalty",
        str(args.itergen_recurrence_penalty),
        "--max-new-tokens",
        str(args.itergen_gsm_max_new_tokens),
        "--output",
        str(output_path),
    ]


def cars_train_command(args: argparse.Namespace, split_file: Path, output_path: Path) -> list[str]:
    cars_cuda_visible_devices = select_cuda_visible_devices(
        requested=args.cars_cuda_visible_devices,
        count=visible_device_count(args.cars_cuda_visible_devices),
        min_free_mib=args.gpu_min_free_mib,
        avoid=args.gpu_avoid_devices,
    )
    return [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "benchmark_gsm_vs_cars.py"),
        "--cars-repo",
        str(args.cars_repo),
        "--output",
        str(output_path),
        "--split-file",
        str(split_file),
        "--split-name",
        "train",
        "--gsm-source-dir",
        str(args.gsm_source_dir),
        "--model-name",
        args.eval_model,
        "--cars-style",
        args.cars_style,
        "--max-attempts-per-example",
        str(args.cars_max_attempts_per_example),
        "--max-new-tokens",
        str(args.gsm_cars_max_new_tokens),
        "--cuda-visible-devices",
        cars_cuda_visible_devices,
    ]


def summarize_baseline(label: str, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": label,
        "benchmark_path": str(path),
        "accuracy": float(payload.get("accuracy", 0.0) or 0.0),
        "syntax_rate": float(payload.get("syntax_rate", 0.0) or 0.0),
        "num_correct": payload.get("num_correct"),
        "num_examples": payload.get("num_examples"),
    }


def crane_train_cache_identity(args: argparse.Namespace, split_file: Path, split_name: str) -> dict[str, Any]:
    return {
        "method": "crane",
        "dataset": "gsm_symbolic",
        "crane_repo": str(args.crane_repo),
        "eval_model": args.eval_model,
        "eval_backend": args.eval_backend,
        "sample_size": split_size(split_file, split_name),
        "eval_max_steps": args.eval_max_steps,
        "eval_step_token_budget": args.eval_step_token_budget,
        "vllm_max_model_len": args.vllm_max_model_len,
        "gsm_source_dir": str(args.gsm_source_dir),
        "gsm_split_file": str(split_file),
        "gsm_split_name": split_name,
        "split_digest": file_digest(split_file),
    }


def itergen_train_cache_identity(args: argparse.Namespace, split_file: Path, split_name: str) -> dict[str, Any]:
    return {
        "method": "itergen",
        "dataset": "gsm",
        "itergen_repo": str(args.itergen_repo),
        "split_file": str(split_file),
        "split_name": split_name,
        "split_digest": file_digest(split_file),
        "model": args.eval_model,
        "seed": args.itergen_seed,
        "recurrence_penalty": args.itergen_recurrence_penalty,
        "max_new_tokens": args.itergen_gsm_max_new_tokens,
    }


def cars_train_cache_identity(args: argparse.Namespace, split_file: Path, split_name: str) -> dict[str, Any]:
    return {
        "method": "cars",
        "dataset": "gsm",
        "cars_repo": str(args.cars_repo),
        "model_name": args.eval_model,
        "split_file": str(split_file),
        "split_name": split_name,
        "split_digest": file_digest(split_file),
        "cars_style": args.cars_style,
        "max_attempts_per_example": args.cars_max_attempts_per_example,
        "max_new_tokens": args.gsm_cars_max_new_tokens,
    }


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    from evaluations.gsm_symbolic.dataset import (
        annotate_gsm_crane_rubric_difficulty_labels,
        write_gsm_stratified_train_eval_split,
        write_gsm_train_eval_split,
    )

    split_file = args.split_file
    if split_file is None:
        split_tag = "stratified" if args.split_strategy == "stratified" else f"train{int(args.train_fraction * 100)}"
        split_file = (
            args.output_dir
            / "splits"
            / f"gsm_symbolic_seed{args.split_seed}_{split_tag}.json"
        )

    difficulty_annotation = None
    if args.annotate_difficulty:
        difficulty_annotation = annotate_gsm_crane_rubric_difficulty_labels(
            crane_dir=args.gsm_source_dir,
            backup=not args.no_difficulty_backup,
        )
        print(
            "[prepare] absolute rubric labels in CRANE folder: "
            f"updated={difficulty_annotation['updated_files']} "
            f"source={difficulty_annotation['difficulty_source']}"
        )

    if args.split_strategy == "stratified":
        split = write_gsm_stratified_train_eval_split(
            split_file,
            crane_dir=args.gsm_source_dir,
            train_counts=parse_difficulty_counts(args.difficulty_train_counts),
            eval_counts=parse_difficulty_counts(args.difficulty_eval_counts),
            seed=args.split_seed,
        )
    else:
        split = write_gsm_train_eval_split(
            split_file,
            crane_dir=args.gsm_source_dir,
            train_fraction=args.train_fraction,
            seed=args.split_seed,
        )

    print(f"[prepare] wrote split: {split_file}")
    print(
        f"[prepare] train={split['train_size']} eval={split['eval_size']} "
        f"total={split['total_examples']} seed={split['seed']}"
    )
    if "train_composition" in split:
        print(
            "[prepare] stratified composition: "
            f"train={split['train_composition']} eval={split['eval_composition']} "
            f"source={split['difficulty_source']}"
        )

    logs_dir = args.output_dir / "logs"
    benchmark_path = args.output_dir / "benchmarks" / f"{args.run_name}_crane_train.json"
    print("[prepare] original CRANE train baseline:")
    crane_train_cmd = crane_benchmark_command(args, split_file, "train", benchmark_path)
    crane_cache = reuse_cached_baseline(
        output_dir=args.output_dir,
        dataset="gsm_symbolic",
        method="crane",
        identity=crane_train_cache_identity(args, split_file, "train"),
        output_path=benchmark_path,
        label="GSM CRANE train",
        validator=accuracy_syntax_validator,
    )
    if not crane_cache["hit"]:
        run_logged(
            crane_train_cmd,
            logs_dir / f"{args.run_name}_crane_train.log",
            env=original_framework_env(args),
        )
        store_cached_baseline(
            output_dir=args.output_dir,
            dataset="gsm_symbolic",
            method="crane",
            identity=crane_train_cache_identity(args, split_file, "train"),
            output_path=benchmark_path,
            label="GSM CRANE train",
            validator=accuracy_syntax_validator,
        )
    train_result = json.loads(benchmark_path.read_text())
    train_result["label"] = "CRANE_train"
    train_result["split_file"] = str(split_file)
    train_result["split_name"] = "train"
    write_json(benchmark_path, train_result)
    baseline_results = [summarize_baseline("CRANE_train", benchmark_path, train_result)]
    print(
        "[prepare] CRANE train: "
        f"accuracy={train_result['accuracy']:.1%} "
        f"({train_result['num_correct']}/{train_result['num_examples']}), "
        f"syntax={train_result['syntax_rate']:.1%}"
    )
    print(f"[prepare] saved benchmark: {benchmark_path}")

    if not args.skip_itergen_train_baseline:
        itergen_path = args.output_dir / "benchmarks" / f"{args.run_name}_itergen_train.json"
        print("[prepare] IterGen train baseline:")
        itergen_cache = reuse_cached_baseline(
            output_dir=args.output_dir,
            dataset="gsm_symbolic",
            method="itergen",
            identity=itergen_train_cache_identity(args, split_file, "train"),
            output_path=itergen_path,
            label="GSM IterGen train",
            validator=accuracy_syntax_validator,
        )
        if not itergen_cache["hit"]:
            run_logged(
                itergen_train_command(args, split_file, itergen_path),
                logs_dir / f"{args.run_name}_itergen_train.log",
                env=original_framework_env(args),
            )
            store_cached_baseline(
                output_dir=args.output_dir,
                dataset="gsm_symbolic",
                method="itergen",
                identity=itergen_train_cache_identity(args, split_file, "train"),
                output_path=itergen_path,
                label="GSM IterGen train",
                validator=accuracy_syntax_validator,
            )
        itergen_result = json.loads(itergen_path.read_text())
        baseline_results.append(summarize_baseline("IterGen_train", itergen_path, itergen_result))

    if not args.skip_cars_train_baseline:
        cars_path = args.output_dir / "benchmarks" / f"{args.run_name}_cars_train.json"
        print("[prepare] CARS train baseline:")
        cars_cache = reuse_cached_baseline(
            output_dir=args.output_dir,
            dataset="gsm_symbolic",
            method="cars",
            identity=cars_train_cache_identity(args, split_file, "train"),
            output_path=cars_path,
            label="GSM CARS train",
            validator=accuracy_syntax_validator,
        )
        if not cars_cache["hit"]:
            run_logged(
                cars_train_command(args, split_file, cars_path),
                logs_dir / f"{args.run_name}_cars_train.log",
            )
            store_cached_baseline(
                output_dir=args.output_dir,
                dataset="gsm_symbolic",
                method="cars",
                identity=cars_train_cache_identity(args, split_file, "train"),
                output_path=cars_path,
                label="GSM CARS train",
                validator=accuracy_syntax_validator,
            )
        cars_result = json.loads(cars_path.read_text())
        baseline_results.append(summarize_baseline("CARS_train", cars_path, cars_result))

    threshold_accuracy_base = max(result["accuracy"] for result in baseline_results)
    threshold_syntax_base = max(result["syntax_rate"] for result in baseline_results)
    train_size = int(split["train_size"])
    min_accuracy = strict_next_threshold(threshold_accuracy_base, train_size)
    min_syntax_rate = strict_next_threshold(threshold_syntax_base, train_size)
    cmd = build_synthesis_command(
        split_file=split_file,
        train_size=train_size,
        min_accuracy=min_accuracy,
        min_syntax_rate=min_syntax_rate,
        args=args,
    )
    launch = {
        "split_file": str(split_file),
        "crane_train_benchmark": str(benchmark_path),
        "train_baselines": baseline_results,
        "crane_framework": "original_crane_repo",
        "crane_train_command": crane_train_cmd,
        "min_accuracy": min_accuracy,
        "min_syntax_rate": min_syntax_rate,
        "threshold_policy": "strict_next_discrete_over_max_train_baseline_or_1_if_saturated",
        "synthesis_command": cmd,
        "synthesis_command_text": command_text(cmd),
        "difficulty_annotation": difficulty_annotation,
    }
    launch_path = args.output_dir / "benchmarks" / f"{args.run_name}_launch.json"
    write_json(launch_path, launch)
    print(
        "[prepare] strict synthesis thresholds: "
        f"accuracy>max_baseline({threshold_accuracy_base:.1%}) => {min_accuracy:.1%}, "
        f"syntax>max_baseline({threshold_syntax_base:.1%}) => {min_syntax_rate:.1%}"
    )
    print(f"[prepare] saved launch metadata: {launch_path}")
    print("[prepare] synthesis command:")
    print(command_text(cmd))
    return launch


def heldout(args: argparse.Namespace) -> dict[str, Any]:
    split_file = args.split_file
    if split_file is None:
        raise ValueError("--split-file is required for heldout")

    csd_module = args.csd_module
    if csd_module is None:
        raise ValueError("--csd-module is required for heldout")

    logs_dir = args.output_dir / "logs"
    crane_eval_path = args.output_dir / "benchmarks" / f"{args.run_name}_crane_eval.json"
    crane_eval_cmd = crane_benchmark_command(args, split_file, "eval", crane_eval_path)
    run_logged(
        crane_eval_cmd,
        logs_dir / f"{args.run_name}_crane_eval.log",
    )
    crane_eval = json.loads(crane_eval_path.read_text())
    crane_eval["label"] = "CRANE_eval"
    crane_eval["split_file"] = str(split_file)
    crane_eval["split_name"] = "eval"

    results = [
        crane_eval,
        evaluate_module(
            module_path=Path(csd_module),
            label=args.csd_label,
            split_file=split_file,
            split_name="eval",
            args=args,
        ),
    ]
    out = {
        "split_file": str(split_file),
        "split_name": "eval",
        "results": results,
        "delta": {
            "accuracy": results[1]["accuracy"] - results[0]["accuracy"],
            "syntax_rate": results[1]["syntax_rate"] - results[0]["syntax_rate"],
        },
    }
    output_path = args.output_dir / "benchmarks" / f"{args.run_name}_heldout_compare.json"
    write_json(output_path, out)

    print("[heldout] CRANE eval: "
          f"accuracy={results[0]['accuracy']:.1%}, syntax={results[0]['syntax_rate']:.1%}")
    print("[heldout] CSD eval: "
          f"accuracy={results[1]['accuracy']:.1%}, syntax={results[1]['syntax_rate']:.1%}")
    print("[heldout] delta: "
          f"accuracy={out['delta']['accuracy']:+.1%}, "
          f"syntax={out['delta']['syntax_rate']:+.1%}")
    print(f"[heldout] saved comparison: {output_path}")
    return out


def annotate_difficulty(args: argparse.Namespace) -> dict[str, Any]:
    from evaluations.gsm_symbolic.dataset import annotate_gsm_crane_rubric_difficulty_labels

    summary = annotate_gsm_crane_rubric_difficulty_labels(
        crane_dir=args.gsm_source_dir,
        backup=not args.no_difficulty_backup,
    )
    output_path = args.output_dir / "benchmarks" / f"{args.run_name}_difficulty_annotation.json"
    write_json(output_path, summary)
    print(
        "[difficulty] CRANE GSM absolute rubric labels: "
        f"updated={summary['updated_files']} "
        f"composition={summary['difficulty_composition']} "
        f"source={summary['difficulty_source']}"
    )
    print(f"[difficulty] saved summary: {output_path}")
    return summary


def annotate_hf_difficulty(args: argparse.Namespace) -> dict[str, Any]:
    from evaluations.gsm_symbolic.dataset import annotate_gsm_crane_hf_difficulty_matches

    summary = annotate_gsm_crane_hf_difficulty_matches(
        crane_dir=args.gsm_source_dir,
        split=args.hf_split,
        overwrite=args.overwrite_difficulty,
        backup=not args.no_difficulty_backup,
    )
    output_path = args.output_dir / "benchmarks" / f"{args.run_name}_hf_difficulty_match.json"
    write_json(output_path, summary)
    print(
        "[hf-difficulty] CRANE GSM HF matches: "
        f"updated={summary['updated_files']} "
        f"label_updates={summary['difficulty_label_updates']} "
        f"stats={summary['match_stats']}"
    )
    print(f"[hf-difficulty] saved summary: {output_path}")
    return summary


def annotate_rubric_difficulty(args: argparse.Namespace) -> dict[str, Any]:
    from evaluations.gsm_symbolic.dataset import annotate_gsm_crane_rubric_difficulty_labels

    summary = annotate_gsm_crane_rubric_difficulty_labels(
        crane_dir=args.gsm_source_dir,
        backup=not args.no_difficulty_backup,
    )
    output_path = args.output_dir / "benchmarks" / f"{args.run_name}_rubric_difficulty.json"
    write_json(output_path, summary)
    print(
        "[rubric-difficulty] CRANE GSM labels: "
        f"updated={summary['updated_files']} "
        f"composition={summary['difficulty_composition']}"
    )
    print(f"[rubric-difficulty] saved summary: {output_path}")
    return summary


def run_synthesis_command(cmd: list[str], log_path: Path) -> Path:
    """Run synthesis, tee output to a log, and return the compiled module path."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("[run-all] launching synthesis:")
    print(command_text(cmd))
    print(f"[run-all] synthesis log: {log_path}")

    compiled_module: Path | None = None
    with log_path.open("w") as log_file:
        process = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            log_file.flush()
            match = re.search(r"^Compiled module:\s*(.+?)\s*$", line)
            if match:
                compiled_module = Path(match.group(1))

        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(
            f"Synthesis command failed with exit code {return_code}. "
            f"See log: {log_path}"
        )
    if compiled_module is None:
        log_text = log_path.read_text(errors="replace")
        matches = re.findall(r"^Compiled module:\s*(.+?)\s*$", log_text, flags=re.MULTILINE)
        if matches:
            compiled_module = Path(matches[-1])
    if compiled_module is None:
        raise RuntimeError(
            "Synthesis completed but no `Compiled module:` line was found. "
            f"See log: {log_path}"
        )

    print(f"[run-all] synthesized module: {compiled_module}")
    return compiled_module


def run_all(args: argparse.Namespace) -> dict[str, Any]:
    """Run split creation, CRANE train benchmark, synthesis, and held-out comparison."""
    launch = prepare(args)
    synthesis_log = args.synthesis_log
    if synthesis_log is None:
        synthesis_log = args.output_dir / "logs" / f"{args.run_name}_synthesis.log"

    compiled_module = run_synthesis_command(
        launch["synthesis_command"],
        synthesis_log,
    )

    args.split_file = Path(launch["split_file"])
    args.csd_module = compiled_module
    args.csd_label = args.csd_label or "CSD_eval"
    comparison = heldout(args)

    summary = {
        "launch": launch,
        "synthesis_log": str(synthesis_log),
        "compiled_module": str(compiled_module),
        "heldout_comparison": comparison,
    }
    summary_path = args.output_dir / "benchmarks" / f"{args.run_name}_run_all_summary.json"
    write_json(summary_path, summary)
    print(f"[run-all] saved full workflow summary: {summary_path}")
    return summary


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-name", default="gsm_split_train50_eval50")
    parser.add_argument(
        "--gsm-source-dir",
        default=str(DEFAULT_CRANE_GSM_DIR),
        help=(
            "Path to the original CRANE GSM-Symbolic JSON folder. Defaults to the "
            "CRANE checkout under this csd-generation repo."
        ),
    )
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--split-seed", type=int, default=123)
    parser.add_argument("--split-strategy", choices=["stratified", "random"], default="stratified")
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument(
        "--difficulty-train-counts",
        default="easy=13,medium=12,hard=25",
        help=(
            "Difficulty counts for the train split. Defaults to half of the "
            "absolute rubric distribution in the 100-example CRANE GSM folder."
        ),
    )
    parser.add_argument(
        "--difficulty-eval-counts",
        default="easy=13,medium=12,hard=25",
        help=(
            "Difficulty counts for the eval split. Defaults to the same label "
            "distribution as train."
        ),
    )
    parser.add_argument("--sample-size", type=int, default=50,
                        help="Legacy random-split sample size; explicit split sizes override it.")
    parser.add_argument("--annotate-difficulty", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite-difficulty", action="store_true")
    parser.add_argument("--no-difficulty-backup", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "generated-csd")
    parser.add_argument("--dafny-path", default=default_dafny_path())
    parser.add_argument("--eval-model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--eval-backend", choices=["huggingface", "vllm"], default="vllm")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--crane-repo", type=Path, default=default_crane_repo())
    parser.add_argument("--crane-device", default=os.environ.get("CRANE_DEVICE", "cuda:0"))
    parser.add_argument("--original-framework-cuda-visible-devices", default=os.environ.get("ORIGINAL_FRAMEWORK_CUDA_VISIBLE_DEVICES", "auto"))
    parser.add_argument("--gpu-min-free-mib", type=int, default=int(os.environ.get("GPU_MIN_FREE_MIB", "12000")))
    parser.add_argument("--gpu-avoid-devices", default=os.environ.get("GPU_AVOID_DEVICES", ""))
    parser.add_argument("--eval-seed", type=int, default=123)
    parser.add_argument("--eval-max-steps", type=int, default=600)
    parser.add_argument("--eval-step-token-budget", type=int, default=1)
    parser.add_argument("--eval-max-seconds-per-example", type=float, default=120.0)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=None)
    parser.add_argument("--vllm-pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--vllm-max-model-len", type=int, default=8192)
    parser.add_argument("--vllm-enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--itergen-repo", type=Path, default=default_itergen_repo())
    parser.add_argument("--itergen-device", default="cuda:0")
    parser.add_argument("--itergen-seed", type=int, default=0)
    parser.add_argument("--itergen-recurrence-penalty", type=float, default=0.3)
    parser.add_argument("--itergen-gsm-max-new-tokens", type=int, default=128)
    parser.add_argument("--cars-repo", type=Path, default=default_cars_repo())
    parser.add_argument("--cars-style", choices=["rs", "ars", "rsft", "cars"], default="cars")
    parser.add_argument("--cars-max-attempts-per-example", type=int, default=2000)
    parser.add_argument("--gsm-cars-max-new-tokens", type=int, default=128)
    parser.add_argument("--cars-cuda-visible-devices", default=os.environ.get("CARS_CUDA_VISIBLE_DEVICES", "auto"))
    parser.add_argument("--skip-itergen-train-baseline", action="store_true")
    parser.add_argument("--skip-cars-train-baseline", action="store_true")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    add_common_args(prepare_parser)
    prepare_parser.add_argument("--task", default=DEFAULT_TASK)
    prepare_parser.add_argument("--max-iterations", type=int, default=100)
    prepare_parser.add_argument("--generation-model", default="gpt-5.4")
    prepare_parser.add_argument("--generation-backend", choices=["huggingface", "vllm", "openai", "anthropic", "gemini"], default="openai")
    prepare_parser.add_argument("--synthesis-output-name", default="gsm_split_train50_synthesis")
    prepare_parser.add_argument("--synthesis-max-tokens", type=int, default=6144)
    prepare_parser.set_defaults(func=prepare)

    run_all_parser = subparsers.add_parser("run-all")
    add_common_args(run_all_parser)
    run_all_parser.add_argument("--task", default=DEFAULT_TASK)
    run_all_parser.add_argument("--max-iterations", type=int, default=100)
    run_all_parser.add_argument("--generation-model", default="gpt-5.4")
    run_all_parser.add_argument("--generation-backend", choices=["huggingface", "vllm", "openai", "anthropic", "gemini"], default="openai")
    run_all_parser.add_argument("--synthesis-output-name", default="gsm_split_train50_synthesis")
    run_all_parser.add_argument("--synthesis-max-tokens", type=int, default=6144)
    run_all_parser.add_argument("--synthesis-log", type=Path, default=None)
    run_all_parser.add_argument("--csd-label", default="CSD_eval")
    run_all_parser.set_defaults(func=run_all)

    heldout_parser = subparsers.add_parser("heldout")
    add_common_args(heldout_parser)
    heldout_parser.add_argument("--csd-module", type=Path, required=True)
    heldout_parser.add_argument("--csd-label", default="CSD_eval")
    heldout_parser.set_defaults(func=heldout)

    annotate_parser = subparsers.add_parser("annotate-difficulty")
    add_common_args(annotate_parser)
    annotate_parser.set_defaults(func=annotate_difficulty)

    annotate_hf_parser = subparsers.add_parser("annotate-hf-difficulty")
    add_common_args(annotate_hf_parser)
    annotate_hf_parser.add_argument("--hf-split", default="test")
    annotate_hf_parser.set_defaults(func=annotate_hf_difficulty)

    annotate_rubric_parser = subparsers.add_parser("annotate-rubric-difficulty")
    add_common_args(annotate_rubric_parser)
    annotate_rubric_parser.set_defaults(func=annotate_rubric_difficulty)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
