#!/usr/bin/env python3
"""CARS-vs-generated-CSD SMILES generalization workflow.

For each selected SMILES class, this script can run the original CARS benchmark,
synthesize a class-specific CSD strategy using 50 feedback attempts, and test
the generated strategy for 100 target samples with the same benchmark harness.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluations.smiles.dataset import SMILES_CLASSES


DEFAULT_TASK_TEMPLATE = (
    "Generate valid, non-exemplar SMILES strings for the {class_name} molecule "
    "class. Use the hidden parser-guided constrained chunk for the SMILES token "
    "sequence and avoid copying prompt exemplars."
)


def normalize_classes(raw: str) -> list[str]:
    classes = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = sorted(set(classes) - set(SMILES_CLASSES))
    if unknown:
        raise ValueError(f"Unknown SMILES class(es): {unknown}")
    return classes


def command_text(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def strict_next_threshold(pass_rate: float | None, n_examples: int) -> float:
    if n_examples <= 0:
        raise ValueError("n_examples must be positive")
    rate = float(pass_rate or 0.0)
    passed = int(round(rate * n_examples))
    return min(1.0, (passed + 1) / n_examples)


def run_logged(cmd: list[str], log_path: Path, *, dry_run: bool, env: dict[str, str] | None = None) -> int:
    print(command_text(cmd))
    if dry_run:
        return 0
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
        return proc.wait()


def compiled_module_from_log(log_path: Path) -> Path:
    text = log_path.read_text(errors="replace")
    matches = re.findall(r"^Compiled module:\s*(.+?)\s*$", text, flags=re.MULTILINE)
    if not matches:
        raise RuntimeError(f"No `Compiled module:` line found in {log_path}")
    return Path(matches[-1])


def synthesis_command(
    args: argparse.Namespace,
    class_name: str,
    *,
    min_accuracy: float,
    min_syntax_rate: float,
) -> list[str]:
    return [
        sys.executable,
        "run_synthesis.py",
        "--task",
        args.task_template.format(class_name=class_name),
        "--dataset",
        "smiles",
        "--smiles-classes",
        class_name,
        "--smiles-samples-per-class",
        str(args.train_samples),
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
        f"{args.synthesis_output_prefix}_{class_name}",
        "--min-accuracy",
        f"{min_accuracy:.12g}",
        "--min-syntax-rate",
        f"{min_syntax_rate:.12g}",
        "--no-require-delimiters",
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


def benchmark_command(
    args: argparse.Namespace,
    *,
    class_name: str,
    compiled_module: Path | None,
    output_dir: Path,
    run_cars: bool,
    run_csd: bool,
    target_samples: int,
) -> list[str]:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "benchmark_smiles_vs_cars.py"),
        "--output-dir",
        str(output_dir),
        "--classes",
        class_name,
        "--model-number",
        args.model_number,
        "--model-name",
        args.eval_model,
        "--backend",
        args.eval_backend,
        "--device",
        args.device,
        "--cars-style",
        args.cars_style,
        "--target-samples",
        str(target_samples),
        "--max-attempts",
        str(args.max_attempts),
        "--max-steps",
        str(args.eval_max_steps),
        "--step-token-budget",
        str(args.eval_step_token_budget),
        "--cuda-visible-devices",
        args.cuda_visible_devices,
    ]
    if args.cars_repo:
        cmd.extend(["--cars-repo", str(args.cars_repo)])
    if compiled_module is not None:
        cmd.extend(["--compiled-module", str(compiled_module)])
    if run_cars:
        cmd.append("--run-cars")
    if run_csd:
        cmd.append("--run-csd")
    if args.dry_run:
        cmd.append("--dry-run")
    return cmd


def itergen_command(
    args: argparse.Namespace,
    *,
    class_name: str,
    output_path: Path,
    target_samples: int,
) -> list[str]:
    return [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_itergen_smiles.py"),
        "--itergen-repo",
        str(args.itergen_repo),
        "--output",
        str(output_path),
        "--classes",
        class_name,
        "--target-samples",
        str(target_samples),
        "--max-attempts",
        str(args.max_attempts),
        "--model",
        args.eval_model,
        "--device",
        args.itergen_device,
        "--seed",
        str(args.itergen_seed),
        "--recurrence-penalty",
        str(args.itergen_recurrence_penalty),
        "--max-new-tokens",
        str(args.itergen_smiles_max_new_tokens),
    ]


def latest_benchmark_json(output_dir: Path) -> Path | None:
    candidates = sorted(output_dir.glob("smiles_benchmark_*.json"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def first_class_metric(path: Path | None, section: str) -> dict[str, Any]:
    if path is None:
        return {}
    payload = json.loads(path.read_text())
    rows = payload.get(section) or []
    return rows[0] if rows else {}


def itergen_class_metric(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    rows = payload.get("classes") or []
    return rows[0] if rows else {}


def summarize_baseline(label: str, path: Path | None, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": label,
        "benchmark_path": str(path) if path else None,
        "accuracy": metrics.get("accuracy"),
        "syntax_rate": metrics.get("syntax_rate"),
        "accuracy_num_correct": metrics.get("accuracy_num_correct"),
        "accuracy_denominator": metrics.get("accuracy_denominator"),
        "invalid_outputs_excluded_from_accuracy": metrics.get("invalid_outputs_excluded_from_accuracy"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "generated-csd")
    parser.add_argument("--run-name", default="smiles_generalization")
    parser.add_argument("--classes", default=",".join(SMILES_CLASSES))
    parser.add_argument("--cars-repo", type=Path, default=None,
                        help="Path to the original CARS repo. Required unless --skip-cars.")
    parser.add_argument("--skip-cars", action="store_true")
    parser.add_argument("--skip-synthesis", action="store_true")
    parser.add_argument("--compiled-module", type=Path, default=None,
                        help="Existing compiled CSD module to test for a single selected class.")
    parser.add_argument("--train-samples", type=int, default=50)
    parser.add_argument("--test-samples", type=int, default=100)
    parser.add_argument("--max-attempts", type=int, default=2000)
    parser.add_argument("--task-template", default=DEFAULT_TASK_TEMPLATE)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--generation-model", default="gpt-5.4")
    parser.add_argument("--generation-backend", choices=["huggingface", "vllm", "openai", "anthropic", "gemini"], default="openai")
    parser.add_argument("--synthesis-output-prefix", default="smiles_train50_synthesis")
    parser.add_argument("--synthesis-max-tokens", type=int, default=6144)
    parser.add_argument("--min-accuracy", type=float, default=None,
                        help="Override synthesized CSD accuracy threshold instead of deriving it from train baselines.")
    parser.add_argument("--min-syntax-rate", type=float, default=None,
                        help="Override synthesized CSD syntax threshold instead of deriving it from train baselines.")
    parser.add_argument("--eval-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--eval-backend", choices=["huggingface", "vllm"], default="vllm")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-max-steps", type=int, default=512)
    parser.add_argument("--eval-step-token-budget", type=int, default=1)
    parser.add_argument("--dafny-path", default="/home/aadivyar/.dotnet/tools/dafny")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--vllm-max-model-len", type=int, default=4096)
    parser.add_argument("--model-number", choices=["1", "2"], default="2")
    parser.add_argument("--cars-style", choices=["rs", "ars", "rsft", "cars"], default="cars")
    parser.add_argument("--cuda-visible-devices", default="1,2")
    parser.add_argument("--itergen-repo", type=Path, default=Path("/home/aadivyar/itergen"))
    parser.add_argument("--itergen-device", default="cuda:0")
    parser.add_argument("--itergen-seed", type=int, default=0)
    parser.add_argument("--itergen-recurrence-penalty", type=float, default=0.3)
    parser.add_argument("--itergen-smiles-max-new-tokens", type=int, default=512)
    parser.add_argument("--skip-itergen-train-baseline", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    classes = normalize_classes(args.classes)
    if args.compiled_module is not None and len(classes) != 1:
        raise SystemExit("--compiled-module can only be reused when exactly one class is selected")
    if not args.skip_cars and not args.cars_repo:
        raise SystemExit("--cars-repo is required to compare against original CARS; pass --skip-cars for CSD-only")

    start = time.time()
    summaries: list[dict[str, Any]] = []
    logs_dir = args.output_dir / "logs"
    benchmarks_dir = args.output_dir / "benchmarks" / args.run_name

    for class_name in classes:
        class_summary: dict[str, Any] = {"class_name": class_name}
        class_benchmark_dir = benchmarks_dir / class_name

        if not args.skip_cars:
            cars_train_cmd = benchmark_command(
                args,
                class_name=class_name,
                compiled_module=None,
                output_dir=class_benchmark_dir / "cars_train50",
                run_cars=True,
                run_csd=False,
                target_samples=args.train_samples,
            )
            print(f"[cars-train:{class_name}]")
            rc = run_logged(cars_train_cmd, logs_dir / f"{args.run_name}_{class_name}_cars_train.log", dry_run=args.dry_run)
            if rc != 0:
                raise SystemExit(rc)
            cars_train_path = latest_benchmark_json(class_benchmark_dir / "cars_train50")
            class_summary["cars_train_benchmark"] = str(cars_train_path)
        else:
            cars_train_path = None

        train_baselines: list[dict[str, Any]] = []
        if not args.skip_cars and cars_train_path is not None and not args.dry_run:
            train_baselines.append(
                summarize_baseline("CARS_train", cars_train_path, first_class_metric(cars_train_path, "cars"))
            )

        if not args.skip_itergen_train_baseline:
            itergen_train_path = class_benchmark_dir / f"itergen_train50_{class_name}.json"
            itergen_train_cmd = itergen_command(
                args,
                class_name=class_name,
                output_path=itergen_train_path,
                target_samples=args.train_samples,
            )
            print(f"[itergen-train:{class_name}]")
            rc = run_logged(itergen_train_cmd, logs_dir / f"{args.run_name}_{class_name}_itergen_train.log", dry_run=args.dry_run)
            if rc != 0:
                raise SystemExit(rc)
            class_summary["itergen_train_benchmark"] = str(itergen_train_path)
            if not args.dry_run:
                train_baselines.append(
                    summarize_baseline("IterGen_train", itergen_train_path, itergen_class_metric(itergen_train_path))
                )

        if args.min_accuracy is None:
            max_accuracy = max((float(b["accuracy"] or 0.0) for b in train_baselines), default=0.0)
            min_accuracy = strict_next_threshold(max_accuracy, args.train_samples)
        else:
            max_accuracy = args.min_accuracy
            min_accuracy = args.min_accuracy
        if args.min_syntax_rate is None:
            max_syntax_rate = max((float(b["syntax_rate"] or 0.0) for b in train_baselines), default=0.0)
            min_syntax_rate = strict_next_threshold(max_syntax_rate, args.train_samples)
        else:
            max_syntax_rate = args.min_syntax_rate
            min_syntax_rate = args.min_syntax_rate
        class_summary["train_baselines"] = train_baselines
        class_summary["threshold_policy"] = "strict_next_discrete_over_max_cars_itergen_train_baseline_or_1_if_saturated"
        class_summary["min_accuracy"] = min_accuracy
        class_summary["min_syntax_rate"] = min_syntax_rate
        print(
            f"[threshold:{class_name}] "
            f"accuracy>max_baseline({max_accuracy:.1%}) => {min_accuracy:.1%}; "
            f"syntax>max_baseline({max_syntax_rate:.1%}) => {min_syntax_rate:.1%}"
        )

        compiled_module = args.compiled_module
        synthesis_log = logs_dir / f"{args.run_name}_{class_name}_synthesis.log"
        if compiled_module is None and not args.skip_synthesis:
            synth_cmd = synthesis_command(
                args,
                class_name,
                min_accuracy=min_accuracy,
                min_syntax_rate=min_syntax_rate,
            )
            print(f"[synthesis:{class_name}]")
            rc = run_logged(synth_cmd, synthesis_log, dry_run=args.dry_run)
            if rc != 0:
                raise SystemExit(rc)
            compiled_module = Path("<dry-run-compiled-module>") if args.dry_run else compiled_module_from_log(synthesis_log)
            class_summary["synthesis_log"] = str(synthesis_log)
            class_summary["compiled_module"] = str(compiled_module)

        if compiled_module is not None:
            if not args.skip_cars:
                cars_test_cmd = benchmark_command(
                    args,
                    class_name=class_name,
                    compiled_module=None,
                    output_dir=class_benchmark_dir / "cars_test100",
                    run_cars=True,
                    run_csd=False,
                    target_samples=args.test_samples,
                )
                print(f"[cars-test:{class_name}]")
                rc = run_logged(cars_test_cmd, logs_dir / f"{args.run_name}_{class_name}_cars_test.log", dry_run=args.dry_run)
                if rc != 0:
                    raise SystemExit(rc)
                class_summary["cars_test_benchmark"] = str(latest_benchmark_json(class_benchmark_dir / "cars_test100"))

            csd_cmd = benchmark_command(
                args,
                class_name=class_name,
                compiled_module=compiled_module,
                output_dir=class_benchmark_dir / "csd",
                run_cars=False,
                run_csd=True,
                target_samples=args.test_samples,
            )
            print(f"[csd-test:{class_name}]")
            rc = run_logged(csd_cmd, logs_dir / f"{args.run_name}_{class_name}_csd_test.log", dry_run=args.dry_run)
            if rc != 0:
                raise SystemExit(rc)
            class_summary["csd_benchmark"] = str(latest_benchmark_json(class_benchmark_dir / "csd"))

        summaries.append(class_summary)

    summary = {
        "config": {
            "classes": classes,
            "train_samples": args.train_samples,
            "test_samples": args.test_samples,
            "cars_repo": str(args.cars_repo) if args.cars_repo else None,
            "dry_run": args.dry_run,
        },
        "classes": summaries,
        "wall_time_seconds": time.time() - start,
    }
    summary_path = args.output_dir / "benchmarks" / f"{args.run_name}_summary.json"
    write_json(summary_path, summary)
    print(f"[summary] wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
