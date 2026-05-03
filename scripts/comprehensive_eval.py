#!/usr/bin/env python3
"""
Comprehensive evaluation runner for the active benchmark suite.

Benchmarks:
- gsm_symbolic
- spider
- smiles

For each selected model and benchmark, this script can:
1. Generate benchmark-specific CSD runs with the current synthesis pipeline.
2. Evaluate each run multiple times.
3. Evaluate an unconstrained baseline for comparison.
4. Save aggregated JSON results.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).parent.parent

MODELS = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
]

BENCHMARKS: dict[str, dict[str, object]] = {
    "gsm_symbolic": {
        "display_name": "GSM-Symbolic",
        "task_description": (
            "Solve math word problems step by step, writing each arithmetic computation "
            "inside << >> delimiters."
        ),
        "eval_module": "evaluations.gsm_symbolic.cli",
        "eval_args": ["--max-steps", "1024"],
        "min_accuracy": 0.30,
        "min_format_rate": 1.0,
        "min_syntax_rate": 1.0,
        "eval_max_steps": 900,
    },
    "spider": {
        "display_name": "Spider",
        "task_description": (
            "Answer text-to-SQL questions by generating a single SQL query from the "
            "provided schema and question."
        ),
        "eval_module": "evaluations.sql_spider.cli",
        "eval_args": ["--max-steps", "400"],
        "min_accuracy": 0.05,
        "min_format_rate": 0.0,
        "min_syntax_rate": 1.0,
        "eval_max_steps": 400,
    },
    "smiles": {
        "display_name": "SMILES",
        "task_description": (
            "Answer constrained molecular generation problems by producing the requested "
            "chemistry answer string, typically a SMILES string, inside << >> delimiters."
        ),
        "eval_module": "evaluations.smiles.cli",
        "eval_args": ["--max-steps", "256"],
        "min_accuracy": 0.20,
        "min_format_rate": 1.0,
        "min_syntax_rate": 1.0,
        "eval_max_steps": 256,
    },
}

NUM_CSDS = 3
NUM_EVAL_RUNS = 3
EVAL_LIMIT = 100


@dataclass
class EvalRunResult:
    run_id: int
    accuracy: float
    format_rate: float
    syntax_rate: float
    avg_tokens: float
    avg_time: float
    total_time: float
    num_examples: int


@dataclass
class CSDEvalResult:
    csd_id: str
    csd_run_dir: str
    runs: List[EvalRunResult] = field(default_factory=list)

    @property
    def avg_accuracy(self) -> float:
        return statistics.mean(r.accuracy for r in self.runs) if self.runs else 0.0

    @property
    def std_accuracy(self) -> float:
        return statistics.stdev(r.accuracy for r in self.runs) if len(self.runs) >= 2 else 0.0

    @property
    def avg_format_rate(self) -> float:
        return statistics.mean(r.format_rate for r in self.runs) if self.runs else 0.0

    @property
    def avg_syntax_rate(self) -> float:
        return statistics.mean(r.syntax_rate for r in self.runs) if self.runs else 0.0

    @property
    def avg_time(self) -> float:
        return statistics.mean(r.avg_time for r in self.runs) if self.runs else 0.0

    def to_dict(self) -> dict:
        return {
            "csd_id": self.csd_id,
            "csd_run_dir": self.csd_run_dir,
            "num_runs": len(self.runs),
            "avg_accuracy": self.avg_accuracy,
            "std_accuracy": self.std_accuracy,
            "avg_format_rate": self.avg_format_rate,
            "avg_syntax_rate": self.avg_syntax_rate,
            "avg_time_per_example": self.avg_time,
            "runs": [asdict(run) for run in self.runs],
        }


@dataclass
class BaselineResults:
    runs: List[EvalRunResult] = field(default_factory=list)

    @property
    def avg_accuracy(self) -> float:
        return statistics.mean(r.accuracy for r in self.runs) if self.runs else 0.0

    @property
    def std_accuracy(self) -> float:
        return statistics.stdev(r.accuracy for r in self.runs) if len(self.runs) >= 2 else 0.0

    @property
    def avg_format_rate(self) -> float:
        return statistics.mean(r.format_rate for r in self.runs) if self.runs else 0.0

    @property
    def avg_syntax_rate(self) -> float:
        return statistics.mean(r.syntax_rate for r in self.runs) if self.runs else 0.0

    @property
    def avg_time(self) -> float:
        return statistics.mean(r.avg_time for r in self.runs) if self.runs else 0.0

    def to_dict(self) -> dict:
        return {
            "num_runs": len(self.runs),
            "avg_accuracy": self.avg_accuracy,
            "std_accuracy": self.std_accuracy,
            "avg_format_rate": self.avg_format_rate,
            "avg_syntax_rate": self.avg_syntax_rate,
            "avg_time_per_example": self.avg_time,
            "runs": [asdict(run) for run in self.runs],
        }


@dataclass
class DatasetResults:
    dataset: str
    csd_results: List[CSDEvalResult] = field(default_factory=list)
    baseline_results: Optional[BaselineResults] = None

    @property
    def best_csd(self) -> Optional[CSDEvalResult]:
        if not self.csd_results:
            return None
        return max(self.csd_results, key=lambda result: result.avg_accuracy)

    @property
    def overall_avg_accuracy(self) -> float:
        return statistics.mean(r.avg_accuracy for r in self.csd_results) if self.csd_results else 0.0

    @property
    def overall_std_accuracy(self) -> float:
        return statistics.stdev(r.avg_accuracy for r in self.csd_results) if len(self.csd_results) >= 2 else 0.0

    @property
    def improvement_over_baseline(self) -> Optional[float]:
        if not self.baseline_results or not self.best_csd:
            return None
        return self.best_csd.avg_accuracy - self.baseline_results.avg_accuracy

    def to_dict(self) -> dict:
        best = self.best_csd
        payload = {
            "dataset": self.dataset,
            "num_csds": len(self.csd_results),
            "overall_avg_accuracy": self.overall_avg_accuracy,
            "overall_std_accuracy": self.overall_std_accuracy,
            "best_csd": {
                "csd_id": best.csd_id if best else None,
                "accuracy": best.avg_accuracy if best else None,
                "std": best.std_accuracy if best else None,
            },
            "csd_results": [result.to_dict() for result in self.csd_results],
        }
        if self.baseline_results:
            payload["baseline"] = self.baseline_results.to_dict()
            payload["comparison"] = {
                "baseline_accuracy": self.baseline_results.avg_accuracy,
                "best_csd_accuracy": best.avg_accuracy if best else None,
                "improvement_best_csd": self.improvement_over_baseline,
            }
        return payload


@dataclass
class ModelResults:
    model_name: str
    benchmark_results: Dict[str, DatasetResults] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "model": self.model_name,
            "benchmarks": {
                name: result.to_dict() for name, result in self.benchmark_results.items()
            },
        }


def run_command(
    cmd: List[str],
    env: Optional[Dict[str, str]] = None,
    timeout: int = 3600,
) -> Tuple[int, str, str]:
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=full_env,
            cwd=str(PROJECT_ROOT),
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except Exception as exc:
        return -1, "", str(exc)


def _display_name(benchmark: str) -> str:
    return str(BENCHMARKS[benchmark]["display_name"])


def _parse_percentage_line(line: str) -> Optional[float]:
    if ":" not in line or "%" not in line:
        return None
    try:
        return float(line.split(":", 1)[1].split("%", 1)[0].strip())
    except Exception:
        return None


def _parse_examples_count(lines: Sequence[str]) -> int:
    for line in lines:
        if line.startswith("Examples:"):
            try:
                return int(line.split(":", 1)[1].strip())
            except Exception:
                pass
        if line.startswith("Examples scored:"):
            try:
                return int(line.split(":", 1)[1].strip())
            except Exception:
                pass
        if "Overall Accuracy:" in line and "(" in line and "/" in line:
            try:
                return int(line.split("(", 1)[1].split("/", 1)[1].split(")", 1)[0].strip())
            except Exception:
                pass
    return 0


def _parse_contains_delimiters(line: str) -> Optional[float]:
    if not line.startswith("Contains << >>:"):
        return None
    if "(" in line and "/" in line:
        try:
            counts = line.split("(", 1)[1].split(")", 1)[0]
            numerator, denominator = counts.split("/", 1)
            return (float(numerator) / float(denominator)) * 100.0
        except Exception:
            pass
    return 100.0 if "yes" in line.lower() else 0.0


def parse_eval_output(stdout: str, benchmark: str, total_time: float) -> Optional[EvalRunResult]:
    lines = [line.strip() for line in stdout.strip().splitlines() if line.strip()]
    num_examples = _parse_examples_count(lines)
    accuracy = 0.0
    format_rate = 0.0
    syntax_rate = 0.0
    avg_tokens = 0.0
    avg_time = 0.0

    accuracy_prefix = {
        "gsm_symbolic": "Answer Accuracy:",
        "spider": "Execution Accuracy (all):",
        "smiles": "Answer Accuracy:",
    }[benchmark]

    for line in lines:
        if line.startswith(accuracy_prefix):
            parsed = _parse_percentage_line(line)
            if parsed is not None:
                accuracy = parsed
        elif line.startswith("Valid Format Rate:"):
            parsed = _parse_percentage_line(line)
            if parsed is not None:
                format_rate = parsed
        elif line.startswith("Syntax Validity:"):
            parsed = _parse_percentage_line(line)
            if parsed is not None:
                syntax_rate = parsed
        elif line.startswith("Avg Tokens:"):
            try:
                avg_tokens = float(line.split(":", 1)[1].strip())
            except Exception:
                pass
        elif line.startswith("Avg Time:"):
            try:
                avg_time = float(line.split(":", 1)[1].strip().rstrip("s"))
            except Exception:
                pass

        contains_rate = _parse_contains_delimiters(line)
        if contains_rate is not None:
            format_rate = contains_rate

    if num_examples == 0:
        return None

    return EvalRunResult(
        run_id=0,
        accuracy=accuracy,
        format_rate=format_rate,
        syntax_rate=syntax_rate,
        avg_tokens=avg_tokens,
        avg_time=avg_time,
        total_time=total_time,
        num_examples=num_examples,
    )


def generate_csd(
    benchmark: str,
    model: str,
    output_name: str,
    temperature: float = 0.7,
    max_iterations: int = 10,
) -> Optional[str]:
    config = BENCHMARKS[benchmark]
    cmd = [
        sys.executable,
        "run_synthesis.py",
        "--task",
        str(config["task_description"]),
        "--dataset",
        benchmark,
        "--output-name",
        output_name,
        "--generation-model",
        model,
        "--eval-model",
        model,
        "--temperature",
        str(temperature),
        "--max-iterations",
        str(max_iterations),
        "--device",
        "auto",
        "--min-accuracy",
        str(config["min_accuracy"]),
        "--min-format-rate",
        str(config["min_format_rate"]),
        "--min-syntax-rate",
        str(config["min_syntax_rate"]),
        "--eval-sample-size",
        "10",
        "--eval-max-steps",
        str(config["eval_max_steps"]),
        "--no-save-reports",
    ]
    if benchmark == "spider":
        cmd.append("--no-require-delimiters")

    print(
        "  Running: python run_synthesis.py --task '...' "
        f"--dataset {benchmark} --output-name {output_name}"
    )
    returncode, stdout, stderr = run_command(cmd, timeout=5400)

    if returncode != 0:
        print(f"  ✗ {_display_name(benchmark)} CSD generation failed")
        print(f"  stderr: {stderr[:500]}")
        return None

    latest_run_file = PROJECT_ROOT / "outputs" / "generated-csd" / "latest_run.txt"
    if latest_run_file.exists():
        run_dir = latest_run_file.read_text().strip()
        print(f"  ✓ CSD generated: {run_dir}")
        return run_dir

    for line in stdout.splitlines():
        if "Run directory:" in line:
            run_dir = line.split("Run directory:", 1)[1].strip()
            if run_dir:
                print(f"  ✓ CSD generated: {run_dir}")
                return run_dir

    print("  ✗ Could not determine run directory")
    return None


def run_benchmark_evaluation(
    benchmark: str,
    run_dir: str,
    model: str,
    limit: int,
    device: str,
    random_sample: bool = True,
    unconstrained: bool = False,
) -> Optional[EvalRunResult]:
    config = BENCHMARKS[benchmark]
    cmd = [
        sys.executable,
        "-m",
        str(config["eval_module"]),
        "--run-dir",
        run_dir,
        "--model",
        model,
        "--device",
        device,
        "--limit",
        str(limit),
    ]
    cmd.extend(str(arg) for arg in config["eval_args"])
    if random_sample:
        cmd.append("--random-sample")
    if unconstrained:
        cmd.append("--unconstrained")

    mode = "baseline (unconstrained)" if unconstrained else "CSD"
    print(f"    Running {_display_name(benchmark)} {mode} evaluation (limit={limit})...")
    start_time = time.time()
    returncode, stdout, stderr = run_command(cmd, timeout=7200)
    total_time = time.time() - start_time

    if returncode != 0:
        print(f"    ✗ {_display_name(benchmark)} evaluation failed")
        print(f"    stderr: {stderr[:500]}")
        return None

    result = parse_eval_output(stdout, benchmark, total_time)
    if result is not None:
        print(
            f"    ✓ Accuracy: {result.accuracy:.1f}% | Format: {result.format_rate:.1f}% | "
            f"Syntax: {result.syntax_rate:.1f}% | Time: {result.avg_time:.2f}s/ex"
        )
    return result


def generate_csds_for_model(
    model: str,
    benchmark: str,
    num_csds: int,
) -> List[str]:
    print(f"\n{'=' * 60}")
    print(f"Generating {_display_name(benchmark)} CSDs for: {model}")
    print(f"Number of CSDs: {num_csds}")
    print(f"{'=' * 60}")

    run_dirs: list[str] = []
    bench_tag = benchmark.replace("_", "-")
    for index in range(num_csds):
        output_name = f"{bench_tag}_csd_{index + 1}_{secrets.token_hex(2)}"
        temperature = 0.7 + (index * 0.1)
        run_dir = generate_csd(
            benchmark=benchmark,
            model=model,
            output_name=output_name,
            temperature=temperature,
        )
        if run_dir:
            run_dirs.append(run_dir)
        else:
            print(f"  ⚠ Failed to generate {_display_name(benchmark)} CSD {index + 1}")
    return run_dirs


def evaluate_model(
    model: str,
    benchmark_run_dirs: Dict[str, List[str]],
    benchmarks: Sequence[str],
    eval_limit: int,
    num_eval_runs: int,
    run_baseline: bool,
    device: str,
) -> ModelResults:
    print(f"\n{'=' * 60}")
    print(f"Evaluating model: {model}")
    for benchmark in benchmarks:
        print(f"{_display_name(benchmark)} CSDs: {len(benchmark_run_dirs.get(benchmark, []))}")
    print(f"Runs per CSD: {num_eval_runs}")
    print(f"Baseline evaluation: {'Yes' if run_baseline else 'No'}")
    print(f"{'=' * 60}")

    model_results = ModelResults(model_name=model)
    global_reference_dir = next(
        (run_dir for benchmark in benchmarks for run_dir in benchmark_run_dirs.get(benchmark, [])),
        None,
    )

    for benchmark in benchmarks:
        print(f"\n--- {_display_name(benchmark)} Evaluation ---")
        run_dirs = benchmark_run_dirs.get(benchmark, [])
        results = DatasetResults(dataset=benchmark)
        reference_dir = run_dirs[0] if run_dirs else global_reference_dir

        if run_baseline and reference_dir:
            print("\n  [BASELINE] Running unconstrained baseline...")
            baseline = BaselineResults()
            for run_idx in range(num_eval_runs):
                print(f"  Baseline Run {run_idx + 1}/{num_eval_runs}:")
                result = run_benchmark_evaluation(
                    benchmark=benchmark,
                    run_dir=reference_dir,
                    model=model,
                    limit=eval_limit,
                    device=device,
                    random_sample=True,
                    unconstrained=True,
                )
                if result:
                    result.run_id = run_idx + 1
                    baseline.runs.append(result)
            if baseline.runs:
                results.baseline_results = baseline
                print(
                    f"  → Baseline avg accuracy: {baseline.avg_accuracy:.1f}% ± "
                    f"{baseline.std_accuracy:.1f}%"
                )

        for csd_idx, run_dir in enumerate(run_dirs):
            csd_id = f"{benchmark}_csd_{csd_idx + 1}"
            print(f"\n  [CSD {csd_idx + 1}/{len(run_dirs)}]: {Path(run_dir).name}")
            csd_result = CSDEvalResult(csd_id=csd_id, csd_run_dir=run_dir)
            for run_idx in range(num_eval_runs):
                print(f"  Run {run_idx + 1}/{num_eval_runs}:")
                result = run_benchmark_evaluation(
                    benchmark=benchmark,
                    run_dir=run_dir,
                    model=model,
                    limit=eval_limit,
                    device=device,
                    random_sample=True,
                    unconstrained=False,
                )
                if result:
                    result.run_id = run_idx + 1
                    csd_result.runs.append(result)
            if csd_result.runs:
                results.csd_results.append(csd_result)
                print(
                    f"  → CSD avg accuracy: {csd_result.avg_accuracy:.1f}% ± "
                    f"{csd_result.std_accuracy:.1f}%"
                )

        if results.baseline_results and results.best_csd:
            improvement = results.improvement_over_baseline
            print(
                f"\n  Summary: Baseline={results.baseline_results.avg_accuracy:.1f}% | "
                f"Best CSD={results.best_csd.avg_accuracy:.1f}% | "
                f"Improvement={improvement:+.1f}%"
            )

        model_results.benchmark_results[benchmark] = results

    return model_results


def print_summary(all_results: Sequence[ModelResults], benchmarks: Sequence[str]) -> None:
    print("\n" + "=" * 80)
    print("COMPREHENSIVE EVALUATION SUMMARY")
    print("=" * 80)

    for model_result in all_results:
        print(f"\n{'=' * 60}")
        print(f"Model: {model_result.model_name}")
        print(f"{'=' * 60}")
        for benchmark in benchmarks:
            result = model_result.benchmark_results.get(benchmark)
            if result is None:
                continue
            print(f"\n  {_display_name(benchmark)}:")
            if result.baseline_results:
                print(
                    f"    Baseline: {result.baseline_results.avg_accuracy:.1f}% ± "
                    f"{result.baseline_results.std_accuracy:.1f}%"
                )
            print(
                f"    CSD Overall Accuracy: {result.overall_avg_accuracy:.1f}% ± "
                f"{result.overall_std_accuracy:.1f}%"
            )
            if result.best_csd:
                print(
                    f"    Best CSD: {result.best_csd.csd_id} "
                    f"({result.best_csd.avg_accuracy:.1f}% ± {result.best_csd.std_accuracy:.1f}%)"
                )
            if result.improvement_over_baseline is not None and result.baseline_results:
                improvement = result.improvement_over_baseline
                baseline = result.baseline_results.avg_accuracy
                relative = (improvement / baseline * 100.0) if baseline > 0 else 0.0
                print(
                    f"    Improvement (Best CSD vs Baseline): {improvement:+.1f}% "
                    f"({relative:+.1f}% relative)"
                )


def _collect_run_dirs(
    args: argparse.Namespace,
    benchmarks: Sequence[str],
) -> Dict[str, List[str]]:
    return {
        "gsm_symbolic": list(args.gsm_csd_dirs or args.csd_dirs or []) if "gsm_symbolic" in benchmarks else [],
        "spider": list(args.spider_csd_dirs or args.csd_dirs or []) if "spider" in benchmarks else [],
        "smiles": list(args.smiles_csd_dirs or args.csd_dirs or []) if "smiles" in benchmarks else [],
    }


def _write_results(
    output_path: str,
    all_results: Sequence[ModelResults],
    args: argparse.Namespace,
    models_to_test: Sequence[str],
    benchmarks: Sequence[str],
    start_time: datetime,
) -> None:
    now = datetime.now()
    payload = {
        "timestamp": now.isoformat(),
        "duration_seconds": (now - start_time).total_seconds(),
        "config": {
            "num_csds": args.num_csds,
            "num_eval_runs": args.num_eval_runs,
            "eval_limit": args.eval_limit,
            "run_baseline": not args.skip_baseline,
            "models": list(models_to_test),
            "benchmarks": list(benchmarks),
        },
        "results": [result.to_dict() for result in all_results],
    }
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Comprehensive CSD evaluation runner")
    parser.add_argument("--model", "-m", type=str, default=None, help="Specific model to test")
    parser.add_argument("--output", "-o", type=str, default="comprehensive_results.json")
    parser.add_argument("--num-csds", type=int, default=NUM_CSDS)
    parser.add_argument("--num-eval-runs", type=int, default=NUM_EVAL_RUNS)
    parser.add_argument("--eval-limit", type=int, default=EVAL_LIMIT)
    parser.add_argument("--benchmarks", nargs="+", choices=sorted(BENCHMARKS.keys()), default=list(BENCHMARKS.keys()))
    parser.add_argument("--skip-synthesis", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--csd-dirs", nargs="+", default=None, help="Shared CSD run dirs for all selected benchmarks")
    parser.add_argument("--gsm-csd-dirs", nargs="+", default=None)
    parser.add_argument("--spider-csd-dirs", nargs="+", default=None)
    parser.add_argument("--smiles-csd-dirs", nargs="+", default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    benchmarks = list(dict.fromkeys(args.benchmarks))
    models_to_test = [args.model] if args.model else MODELS
    run_baseline = not args.skip_baseline

    print("=" * 80)
    print("COMPREHENSIVE CSD EVALUATION")
    print("=" * 80)
    print(f"Models: {len(models_to_test)}")
    print(f"Benchmarks: {', '.join(benchmarks)}")
    print(f"CSDs per benchmark per model: {args.num_csds}")
    print(f"Eval runs per CSD: {args.num_eval_runs}")
    print(f"Examples per eval: {args.eval_limit}")
    print(f"Baseline evaluation: {'Yes' if run_baseline else 'No (skipped)'}")
    benchmark_count = len(benchmarks)
    baseline_runs = args.num_eval_runs * benchmark_count if run_baseline else 0
    csd_runs = args.num_csds * args.num_eval_runs * benchmark_count
    print(f"Total evaluations per model: {baseline_runs + csd_runs}")
    print(f"Output file: {args.output}")
    print("=" * 80)

    all_results: list[ModelResults] = []
    start_time = datetime.now()

    for model in models_to_test:
        print(f"\n\n{'#' * 80}")
        print(f"# Processing: {model}")
        print(f"{'#' * 80}")

        if args.skip_synthesis:
            benchmark_run_dirs = _collect_run_dirs(args, benchmarks)
            if not any(benchmark_run_dirs.get(benchmark) for benchmark in benchmarks):
                print(
                    "Error: --skip-synthesis requires --csd-dirs or one of "
                    "--gsm-csd-dirs / --spider-csd-dirs / --smiles-csd-dirs"
                )
                sys.exit(1)
        else:
            benchmark_run_dirs = {}
            for benchmark in benchmarks:
                print(f"\n  Generating {args.num_csds} {_display_name(benchmark)} CSDs...")
                benchmark_run_dirs[benchmark] = generate_csds_for_model(
                    model=model,
                    benchmark=benchmark,
                    num_csds=args.num_csds,
                )
            if not any(benchmark_run_dirs.get(benchmark) for benchmark in benchmarks):
                print(f"⚠ No CSDs generated for {model}, skipping evaluation")
                continue

        model_results = evaluate_model(
            model=model,
            benchmark_run_dirs=benchmark_run_dirs,
            benchmarks=benchmarks,
            eval_limit=args.eval_limit,
            num_eval_runs=args.num_eval_runs,
            run_baseline=run_baseline,
            device=args.device,
        )
        all_results.append(model_results)
        _write_results(args.output, all_results, args, models_to_test, benchmarks, start_time)
        print(f"\n✓ Intermediate results saved to {args.output}")

    print_summary(all_results, benchmarks)
    _write_results(args.output, all_results, args, models_to_test, benchmarks, start_time)
    end_time = datetime.now()
    print(f"\n✓ Final results saved to {args.output}")
    print(f"Total duration: {(end_time - start_time).total_seconds() / 3600:.1f} hours")


if __name__ == "__main__":
    main()
