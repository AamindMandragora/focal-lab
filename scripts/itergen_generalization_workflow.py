#!/usr/bin/env python3
"""IterGen-vs-generated-CSD Spider generalization workflow.

The workflow creates a non-overlapping Spider split, benchmarks original
IterGen on the 50-example train split, synthesizes a CSD strategy against that
same split, then evaluates IterGen and the generated CSD on a 100-example test
split.
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


DEFAULT_TASK = (
    "Generate a Spider SQL query that answers the natural-language question "
    "using only the provided database schema. Keep SQL generation inside the "
    "hidden constrained parser-guided chunk."
)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def command_text(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def strict_next_threshold(pass_rate: float, n_examples: int) -> float:
    if n_examples <= 0:
        raise ValueError("n_examples must be positive")
    passed = int(round(pass_rate * n_examples))
    return min(1.0, (passed + 1) / n_examples)


def run_logged(cmd: list[str], log_path: Path, *, dry_run: bool) -> int:
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
            env=os.environ.copy(),
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


def split_size(split_file: Path, split_name: str) -> int:
    manifest = json.loads(split_file.read_text())
    return len(manifest[f"{split_name}_indices"])


def make_split(args: argparse.Namespace) -> dict[str, Any]:
    from evaluations.sql_spider.dataset import write_spider_train_test_split

    if args.split_file is None:
        args.split_file = (
            args.output_dir
            / "splits"
            / f"spider_seed{args.split_seed}_train{args.train_size}_test{args.test_size}.json"
        )
    if args.split_file.exists() and not args.regenerate_split:
        split = json.loads(args.split_file.read_text())
    else:
        split = write_spider_train_test_split(
            args.split_file,
            source=args.spider_source,
            spider_dir=args.spider_dir,
            train_size=args.train_size,
            test_size=args.test_size,
            seed=args.split_seed,
        )
    print(
        f"[split] {args.split_file} train={split['train_size']} "
        f"test={split['test_size']} seed={split['seed']}"
    )
    return split


def itergen_command(args: argparse.Namespace, split_name: str, output_path: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_itergen_sql_split.py"),
        "--itergen-repo",
        str(args.itergen_repo),
        "--split-file",
        str(args.split_file),
        "--split-name",
        split_name,
        "--model",
        args.itergen_model,
        "--device",
        args.itergen_device,
        "--seed",
        str(args.itergen_seed),
        "--recurrence-penalty",
        str(args.recurrence_penalty),
        "--max-iter",
        str(args.itergen_max_iter),
        "--source",
        args.spider_source,
        "--output",
        str(output_path),
    ]
    if args.spider_dir is not None:
        cmd.extend(["--spider-dir", str(args.spider_dir)])
    return cmd


def cars_command(args: argparse.Namespace, split_name: str, output_path: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "benchmark_spider_vs_cars.py"),
        "--cars-repo",
        str(args.cars_repo),
        "--output",
        str(output_path),
        "--split-file",
        str(args.split_file),
        "--split-name",
        split_name,
        "--source",
        args.spider_source,
        "--model-name",
        args.eval_model,
        "--cars-style",
        args.cars_style,
        "--max-attempts-per-example",
        str(args.cars_max_attempts_per_example),
        "--max-new-tokens",
        str(args.spider_cars_max_new_tokens),
        "--cuda-visible-devices",
        args.cars_cuda_visible_devices,
    ]
    if args.spider_dir is not None:
        cmd.extend(["--spider-dir", str(args.spider_dir)])
    return cmd


def spider_exec_accuracy(path: Path) -> float:
    payload = json.loads(path.read_text())
    if "all_exec_accuracy" in payload:
        return float(payload.get("all_exec_accuracy") or 0.0)
    return float(payload.get("scores", {}).get("all", {}).get("exec", 0.0) or 0.0)


def synthesis_command(args: argparse.Namespace, min_accuracy: float) -> list[str]:
    return [
        sys.executable,
        "run_synthesis.py",
        "--task",
        args.task,
        "--dataset",
        "spider",
        "--spider-split-file",
        str(args.split_file),
        "--spider-split-name",
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
        f"{args.min_syntax_rate:.12g}",
        "--no-require-delimiters",
        "--eval-sample-size",
        str(split_size(args.split_file, "train")),
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


def csd_test_command(args: argparse.Namespace, compiled_module: Path, output_path: Path) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "evaluations.sql_spider.cli",
        "--run-dir",
        str(compiled_module),
        "--model",
        args.eval_model,
        "--backend",
        args.eval_backend,
        "--device",
        args.device,
        "--split-file",
        str(args.split_file),
        "--split-name",
        "test",
        "--source",
        args.spider_source,
        "--limit",
        str(split_size(args.split_file, "test")),
        "--max-steps",
        str(args.eval_max_steps),
        "--vllm-gpu-memory-utilization",
        str(args.vllm_gpu_memory_utilization),
        "--vllm-max-model-len",
        str(args.vllm_max_model_len),
        "--pred-dump",
        str(output_path),
    ]
    if args.spider_dir is not None:
        cmd.extend(["--spider-dir", str(args.spider_dir)])
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "generated-csd")
    parser.add_argument("--run-name", default="itergen_spider_generalization")
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--split-seed", type=int, default=123)
    parser.add_argument("--train-size", type=int, default=50)
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--regenerate-split", action="store_true")
    parser.add_argument("--spider-source", choices=["auto", "hf", "local"], default="local")
    parser.add_argument("--spider-dir", type=Path, default=None)
    parser.add_argument("--itergen-repo", type=Path, default=Path("/home/aadivyar/itergen"))
    parser.add_argument("--itergen-model", default="Qwen/Qwen2.5-Coder-14B-Instruct")
    parser.add_argument("--itergen-device", default="cuda:0")
    parser.add_argument("--itergen-seed", type=int, default=0)
    parser.add_argument("--recurrence-penalty", type=float, default=0.3)
    parser.add_argument("--itergen-max-iter", type=int, default=20)
    parser.add_argument("--cars-repo", type=Path, default=Path("/home/aadivyar/cars"))
    parser.add_argument("--cars-style", choices=["rs", "ars", "rsft", "cars"], default="cars")
    parser.add_argument("--cars-max-attempts-per-example", type=int, default=2000)
    parser.add_argument("--spider-cars-max-new-tokens", type=int, default=512)
    parser.add_argument("--cars-cuda-visible-devices", default=os.environ.get("CARS_CUDA_VISIBLE_DEVICES", "1,3"))
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--generation-model", default="gpt-5.4")
    parser.add_argument("--generation-backend", choices=["huggingface", "vllm", "openai", "anthropic", "gemini"], default="openai")
    parser.add_argument("--synthesis-output-name", default="itergen_spider_train50_synthesis")
    parser.add_argument("--synthesis-max-tokens", type=int, default=6144)
    parser.add_argument("--eval-model", default="Qwen/Qwen2.5-Coder-14B-Instruct")
    parser.add_argument("--eval-backend", choices=["huggingface", "vllm"], default="vllm")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-max-steps", type=int, default=400)
    parser.add_argument("--eval-step-token-budget", type=int, default=4)
    parser.add_argument("--min-syntax-rate", type=float, default=0.0)
    parser.add_argument("--dafny-path", default="/home/aadivyar/.dotnet/tools/dafny")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--vllm-max-model-len", type=int, default=4096)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    start = time.time()
    split = make_split(args)
    benchmarks_dir = args.output_dir / "benchmarks"
    logs_dir = args.output_dir / "logs"
    train_itergen_json = benchmarks_dir / f"{args.run_name}_itergen_train.json"
    test_itergen_json = benchmarks_dir / f"{args.run_name}_itergen_test.json"
    train_cars_json = benchmarks_dir / f"{args.run_name}_cars_train.json"
    synthesis_log = logs_dir / f"{args.run_name}_synthesis.log"
    csd_test_json = benchmarks_dir / f"{args.run_name}_csd_test.json"

    train_cmd = itergen_command(args, "train", train_itergen_json)
    print("[itergen-train]")
    rc = run_logged(train_cmd, logs_dir / f"{args.run_name}_itergen_train.log", dry_run=args.dry_run)
    if rc != 0:
        raise SystemExit(rc)

    if args.dry_run:
        train_accuracy = 0.0
        cars_train_accuracy = 0.0
        min_accuracy = 0.0
    else:
        train_accuracy = spider_exec_accuracy(train_itergen_json)

        print("[cars-train]")
        cars_train_cmd = cars_command(args, "train", train_cars_json)
        rc = run_logged(cars_train_cmd, logs_dir / f"{args.run_name}_cars_train.log", dry_run=args.dry_run)
        if rc != 0:
            raise SystemExit(rc)
        cars_train_accuracy = spider_exec_accuracy(train_cars_json)

        threshold_base_accuracy = max(train_accuracy, cars_train_accuracy)
        min_accuracy = strict_next_threshold(threshold_base_accuracy, split["train_size"])
    print(
        "[benchmark] train accuracy: "
        f"IterGen={train_accuracy:.1%}, CARS={cars_train_accuracy:.1%}; "
        f"synthesis threshold={min_accuracy:.1%}"
    )

    synth_cmd = synthesis_command(args, min_accuracy)
    print("[synthesis]")
    rc = run_logged(synth_cmd, synthesis_log, dry_run=args.dry_run)
    if rc != 0:
        raise SystemExit(rc)

    compiled_module = Path("<dry-run-compiled-module>")
    if not args.dry_run:
        compiled_module = compiled_module_from_log(synthesis_log)
        print(f"[synthesis] compiled module: {compiled_module}")

    test_itergen_cmd = itergen_command(args, "test", test_itergen_json)
    print("[itergen-test]")
    rc = run_logged(test_itergen_cmd, logs_dir / f"{args.run_name}_itergen_test.log", dry_run=args.dry_run)
    if rc != 0:
        raise SystemExit(rc)

    csd_cmd = csd_test_command(args, compiled_module, csd_test_json)
    print("[csd-test]")
    rc = run_logged(csd_cmd, logs_dir / f"{args.run_name}_csd_test.log", dry_run=args.dry_run)
    if rc != 0:
        raise SystemExit(rc)

    summary: dict[str, Any] = {
        "split_file": str(args.split_file),
        "split": split,
        "itergen_train": str(train_itergen_json),
        "cars_train": str(train_cars_json),
        "itergen_test": str(test_itergen_json),
        "synthesis_log": str(synthesis_log),
        "compiled_module": str(compiled_module),
        "csd_test": str(csd_test_json),
        "commands": {
            "itergen_train": train_cmd,
            "cars_train": cars_command(args, "train", train_cars_json),
            "synthesis": synth_cmd,
            "itergen_test": test_itergen_cmd,
            "csd_test": csd_cmd,
        },
        "threshold_policy": "strict_next_discrete_over_max_itergen_cars_train_exec_accuracy_or_1_if_saturated",
        "dry_run": args.dry_run,
        "wall_time_seconds": time.time() - start,
    }
    if not args.dry_run:
        itergen_test = json.loads(test_itergen_json.read_text())
        csd_test = json.loads(csd_test_json.read_text())
        summary["results"] = {
            "itergen_train_accuracy": train_accuracy,
            "cars_train_accuracy": cars_train_accuracy,
            "itergen_test_accuracy": float(itergen_test.get("all_exec_accuracy", 0.0)),
            "csd_test_accuracy": float(csd_test.get("scores", {}).get("all", {}).get("exec", 0.0)),
        }
    summary_path = benchmarks_dir / f"{args.run_name}_summary.json"
    write_json(summary_path, summary)
    print(f"[summary] wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
