#!/usr/bin/env python3
"""Master experiment matrix runner for GSM, Spider, and SMILES.

Runs the paper-facing matrix over:
  datasets: gsm, spider, smiles
  methods: itergen, cars, metadecode
  eval models: the CRANE table models except QwQ-32B

Unsupported cells are written explicitly to the JSONL ledger.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    name: str


PAPER_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec("qwen25_1p5b_instruct", "Qwen/Qwen2.5-1.5B-Instruct"),
    ModelSpec("qwen25_coder_7b_instruct", "Qwen/Qwen2.5-Coder-7B-Instruct"),
    ModelSpec("qwen25_math_7b_instruct", "Qwen/Qwen2.5-Math-7B-Instruct"),
    ModelSpec("llama31_8b_instruct", "meta-llama/Llama-3.1-8B-Instruct"),
    ModelSpec("deepseek_r1_distill_qwen_7b", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"),
    ModelSpec("deepseek_r1_distill_llama_8b", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"),
    ModelSpec("qwen25_coder_14b_instruct", "Qwen/Qwen2.5-Coder-14B-Instruct"),
    ModelSpec("deepseek_r1_distill_qwen_14b", "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"),
)

DATASETS = ("gsm", "spider", "smiles")
METHODS = ("itergen", "cars", "metadecode")


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def command_text(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def parse_csv(raw: str, allowed: tuple[str, ...], label: str) -> list[str]:
    raw = raw.strip()
    selected = list(allowed) if raw == "all" else [part.strip() for part in raw.split(",") if part.strip()]
    unknown = sorted(set(selected) - set(allowed))
    if unknown:
        raise SystemExit(f"Unknown {label}: {unknown}; allowed={allowed}")
    return selected


def select_models(raw: str) -> list[ModelSpec]:
    if raw.strip() == "all":
        return list(PAPER_MODELS)
    by_alias = {model.alias: model for model in PAPER_MODELS}
    by_name = {model.name: model for model in PAPER_MODELS}
    selected: list[ModelSpec] = []
    unknown: list[str] = []
    for part in [p.strip() for p in raw.split(",") if p.strip()]:
        if part in by_alias:
            selected.append(by_alias[part])
        elif part in by_name:
            selected.append(by_name[part])
        else:
            unknown.append(part)
    if unknown:
        raise SystemExit(
            "Unknown model selector(s): "
            + ", ".join(unknown)
            + ". Use one of: "
            + ", ".join(model.alias for model in PAPER_MODELS)
        )
    return selected


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def append_ledger(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"ts": now_iso(), **row}
    with path.open("a") as f:
        f.write(json.dumps(row, default=str) + "\n")


def status_paths(args: argparse.Namespace) -> dict[str, Path]:
    root = args.output_dir / "master_experiments" / args.run_name
    return {
        "root": root,
        "logs": root / "logs",
        "ledger": root / "matrix_ledger.jsonl",
        "manifest": root / "matrix_manifest.json",
        "latest": args.output_dir / "logs" / "master_experiments_latest.json",
    }


def gpu_env(base: dict[str, str], cuda_visible_devices: str | None) -> dict[str, str]:
    env = dict(base)
    if cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    env.setdefault("PYTHONPATH", str(PROJECT_ROOT))
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    return env


def run_logged(
    *,
    args: argparse.Namespace,
    ledger: Path,
    cell_ids: list[str],
    group_id: str,
    cmd: list[str],
    log_path: Path,
    env: dict[str, str],
    meta: dict[str, Any],
) -> int:
    for cell_id in cell_ids:
        append_ledger(
            ledger,
            {
                "event": "cell_start",
                "cell_id": cell_id,
                "group_id": group_id,
                "status": "running",
                "command": cmd,
                "command_text": command_text(cmd),
                "log_path": str(log_path),
                **meta,
            },
        )
    print(f"[START] {group_id}")
    print(command_text(cmd))
    if args.dry_run:
        rc = 0
    else:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as log_file:
            proc = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="")
                log_file.write(line)
                log_file.flush()
            rc = proc.wait()
    status = "completed" if rc == 0 else "failed"
    for cell_id in cell_ids:
        append_ledger(
            ledger,
            {
                "event": "cell_end",
                "cell_id": cell_id,
                "group_id": group_id,
                "status": status,
                "returncode": rc,
                "log_path": str(log_path),
                **meta,
            },
        )
    print(f"[END] {group_id} rc={rc}")
    if rc != 0 and not args.continue_on_error:
        raise SystemExit(rc)
    return rc


def ensure_spider_split(args: argparse.Namespace) -> None:
    if args.dry_run and args.spider_split_file.exists():
        return
    if args.spider_split_file.exists() and not args.regenerate_splits:
        return
    from evaluations.sql_spider.dataset import write_spider_train_test_split

    split = write_spider_train_test_split(
        args.spider_split_file,
        source=args.spider_source,
        spider_dir=args.spider_dir,
        train_size=args.spider_train_size,
        test_size=args.spider_test_size,
        seed=args.split_seed,
    )
    print(
        f"[split] wrote {args.spider_split_file} "
        f"train={split['train_size']} test={split['test_size']}"
    )


def gsm_metadecode_command(args: argparse.Namespace, model: ModelSpec) -> list[str]:
    return [
        args.python,
        "scripts/gsm_split_synthesis_workflow.py",
        "run-all",
        "--run-name",
        f"{args.run_name}_gsm_metadecode_{model.alias}",
        "--split-file",
        str(args.gsm_split_file),
        "--split-strategy",
        "stratified",
        "--difficulty-train-counts",
        args.gsm_difficulty_train_counts,
        "--difficulty-eval-counts",
        args.gsm_difficulty_eval_counts,
        "--gsm-source-dir",
        str(args.gsm_source_dir),
        "--output-dir",
        str(args.output_dir),
        "--eval-model",
        model.name,
        "--eval-backend",
        args.eval_backend,
        "--device",
        args.device,
        "--max-iterations",
        str(args.max_iterations),
        "--generation-model",
        args.generation_model,
        "--generation-backend",
        args.generation_backend,
        "--eval-max-steps",
        str(args.gsm_eval_max_steps),
        "--eval-step-token-budget",
        str(args.gsm_eval_step_token_budget),
        "--vllm-gpu-memory-utilization",
        str(args.gsm_vllm_gpu_memory_utilization),
        "--vllm-max-model-len",
        str(args.gsm_vllm_max_model_len),
        "--synthesis-max-tokens",
        str(args.synthesis_max_tokens),
    ]


def gsm_itergen_command(args: argparse.Namespace, model: ModelSpec, output_path: Path) -> list[str]:
    return [
        args.python,
        "scripts/run_itergen_gsm_split.py",
        "--itergen-repo",
        str(args.itergen_repo),
        "--split-file",
        str(args.gsm_split_file),
        "--split-name",
        "eval",
        "--gsm-source-dir",
        str(args.gsm_source_dir),
        "--model",
        model.name,
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


def gsm_cars_command(args: argparse.Namespace, model: ModelSpec, output_path: Path) -> list[str]:
    return [
        args.python,
        "scripts/benchmark_gsm_vs_cars.py",
        "--cars-repo",
        str(args.cars_repo),
        "--output",
        str(output_path),
        "--split-file",
        str(args.gsm_split_file),
        "--split-name",
        "eval",
        "--gsm-source-dir",
        str(args.gsm_source_dir),
        "--model-name",
        model.name,
        "--cars-style",
        args.cars_style,
        "--max-attempts-per-example",
        str(args.cars_max_attempts_per_example),
        "--max-new-tokens",
        str(args.gsm_cars_max_new_tokens),
        "--cuda-visible-devices",
        args.cars_cuda_visible_devices,
    ]


def spider_pair_command(args: argparse.Namespace, model: ModelSpec) -> list[str]:
    return [
        args.python,
        "scripts/itergen_generalization_workflow.py",
        "--run-name",
        f"{args.run_name}_spider_itergen_metadecode_{model.alias}",
        "--output-dir",
        str(args.output_dir),
        "--itergen-repo",
        str(args.itergen_repo),
        "--split-file",
        str(args.spider_split_file),
        "--train-size",
        str(args.spider_train_size),
        "--test-size",
        str(args.spider_test_size),
        "--itergen-model",
        model.name,
        "--eval-model",
        model.name,
        "--eval-backend",
        args.eval_backend,
        "--device",
        args.device,
        "--itergen-device",
        args.itergen_device,
        "--itergen-seed",
        str(args.itergen_seed),
        "--recurrence-penalty",
        str(args.itergen_recurrence_penalty),
        "--max-iterations",
        str(args.max_iterations),
        "--generation-model",
        args.generation_model,
        "--generation-backend",
        args.generation_backend,
        "--eval-max-steps",
        str(args.spider_eval_max_steps),
        "--eval-step-token-budget",
        str(args.spider_eval_step_token_budget),
        "--vllm-gpu-memory-utilization",
        str(args.spider_vllm_gpu_memory_utilization),
        "--vllm-max-model-len",
        str(args.spider_vllm_max_model_len),
        "--synthesis-max-tokens",
        str(args.synthesis_max_tokens),
    ]


def spider_cars_command(args: argparse.Namespace, model: ModelSpec, output_path: Path) -> list[str]:
    return [
        args.python,
        "scripts/benchmark_spider_vs_cars.py",
        "--cars-repo",
        str(args.cars_repo),
        "--output",
        str(output_path),
        "--split-file",
        str(args.spider_split_file),
        "--split-name",
        "test",
        "--source",
        args.spider_source,
        "--model-name",
        model.name,
        "--cars-style",
        args.cars_style,
        "--max-attempts-per-example",
        str(args.cars_max_attempts_per_example),
        "--max-new-tokens",
        str(args.spider_cars_max_new_tokens),
        "--cuda-visible-devices",
        args.cars_cuda_visible_devices,
    ]


def smiles_pair_command(args: argparse.Namespace, model: ModelSpec) -> list[str]:
    return [
        args.python,
        "scripts/smiles_generalization_workflow.py",
        "--run-name",
        f"{args.run_name}_smiles_cars_metadecode_{model.alias}",
        "--output-dir",
        str(args.output_dir),
        "--cars-repo",
        str(args.cars_repo),
        "--classes",
        args.smiles_classes,
        "--train-samples",
        str(args.smiles_train_samples),
        "--test-samples",
        str(args.smiles_test_samples),
        "--eval-model",
        model.name,
        "--eval-backend",
        args.eval_backend,
        "--device",
        args.device,
        "--cuda-visible-devices",
        args.smiles_cuda_visible_devices,
        "--model-number",
        "2",
        "--cars-style",
        args.cars_style,
        "--max-attempts",
        str(args.smiles_max_attempts),
        "--max-iterations",
        str(args.max_iterations),
        "--generation-model",
        args.generation_model,
        "--generation-backend",
        args.generation_backend,
        "--eval-max-steps",
        str(args.smiles_eval_max_steps),
        "--eval-step-token-budget",
        str(args.smiles_eval_step_token_budget),
        "--vllm-gpu-memory-utilization",
        str(args.smiles_vllm_gpu_memory_utilization),
        "--vllm-max-model-len",
        str(args.smiles_vllm_max_model_len),
        "--synthesis-max-tokens",
        str(args.synthesis_max_tokens),
    ]


def smiles_itergen_command(args: argparse.Namespace, model: ModelSpec, output_path: Path) -> list[str]:
    return [
        args.python,
        "scripts/run_itergen_smiles.py",
        "--itergen-repo",
        str(args.itergen_repo),
        "--output",
        str(output_path),
        "--classes",
        args.smiles_classes,
        "--target-samples",
        str(args.smiles_test_samples),
        "--max-attempts",
        str(args.smiles_max_attempts),
        "--model",
        model.name,
        "--device",
        args.itergen_device,
        "--seed",
        str(args.itergen_seed),
        "--recurrence-penalty",
        str(args.itergen_recurrence_penalty),
        "--max-new-tokens",
        str(args.itergen_smiles_max_new_tokens),
    ]


def legacy_ablation_commands(args: argparse.Namespace) -> list[tuple[str, list[str], str]]:
    if not args.include_ablations:
        return []
    commands: list[tuple[str, list[str], str]] = []
    commands.append((
        "ablation_gsm_regression",
        ["bash", "scripts/run_gsm_regression.sh"],
        "Existing GSM low-iteration regression synthesis ablation.",
    ))
    if args.include_lottery_ablation:
        commands.append((
            "ablation_gsm_lottery",
            ["bash", "scripts/run_gsm_lottery.sh"],
            "Existing GSM lottery-seed synthesis ablation.",
        ))
    return commands


def build_manifest(args: argparse.Namespace, models: list[ModelSpec], datasets: list[str], methods: list[str]) -> dict[str, Any]:
    cells = []
    for dataset in datasets:
        for method in methods:
            for model in models:
                cells.append({
                    "cell_id": f"{dataset}_{method}_{model.alias}",
                    "dataset": dataset,
                    "method": method,
                    "model_alias": model.alias,
                    "model_name": model.name,
                    "status": "pending",
                })
    return {
        "run_name": args.run_name,
        "created_at": now_iso(),
        "datasets": datasets,
        "methods": methods,
        "models": [model.__dict__ for model in models],
        "excluded_models": ["QwQ-32B"],
        "adapter_contract": (
            "Each dataset supplies its grammar, prompts/examples, and scorer; "
            "each method runner consumes the dataset grammar explicitly."
        ),
        "sizes": {
            "gsm": {
                "train": 50,
                "eval": 50,
                "split_file": str(args.gsm_split_file),
                "difficulty_train_counts": args.gsm_difficulty_train_counts,
                "difficulty_eval_counts": args.gsm_difficulty_eval_counts,
            },
            "spider": {
                "train": args.spider_train_size,
                "test": args.spider_test_size,
                "split_file": str(args.spider_split_file),
            },
            "smiles": {
                "train_samples": args.smiles_train_samples,
                "test_samples": args.smiles_test_samples,
                "classes": args.smiles_classes,
            },
        },
        "cells": cells,
        "include_ablations": args.include_ablations,
        "include_lottery_ablation": args.include_lottery_ablation,
        "dry_run": args.dry_run,
    }


def main() -> int:
    default_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default=f"master_experiments_{default_stamp}")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "generated-csd")
    parser.add_argument("--python", default="/opt/anaconda/bin/python")
    parser.add_argument("--models", default="all", help="all, or comma-separated model aliases/full HF IDs")
    parser.add_argument("--datasets", default="all", help="all, or comma-separated gsm,spider,smiles")
    parser.add_argument("--methods", default="all", help="all, or comma-separated itergen,cars,metadecode")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--regenerate-splits", action="store_true")
    parser.add_argument("--include-ablations", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-lottery-ablation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--generation-model", default="gpt-5.4")
    parser.add_argument("--generation-backend", choices=["huggingface", "vllm", "openai"], default="openai")
    parser.add_argument("--eval-backend", choices=["huggingface", "vllm"], default="vllm")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--synthesis-max-tokens", type=int, default=6144)
    parser.add_argument("--split-seed", type=int, default=123)

    parser.add_argument("--gsm-source-dir", type=Path, default=Path("/home/aadivyar/CRANE/src/gsm_symbolic"))
    parser.add_argument(
        "--gsm-split-file",
        type=Path,
        default=PROJECT_ROOT / "outputs/generated-csd/splits/gsm_absolute_rubric_seed123_train50_eval50.json",
    )
    parser.add_argument("--gsm-difficulty-train-counts", default="easy=13,medium=12,hard=25")
    parser.add_argument("--gsm-difficulty-eval-counts", default="easy=13,medium=12,hard=25")
    parser.add_argument("--gsm-eval-max-steps", type=int, default=600)
    parser.add_argument("--gsm-eval-step-token-budget", type=int, default=1)
    parser.add_argument("--gsm-vllm-gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--gsm-vllm-max-model-len", type=int, default=8192)
    parser.add_argument("--gsm-cuda-visible-devices", default=os.environ.get("GSM_CUDA_VISIBLE_DEVICES", "3"))

    parser.add_argument(
        "--spider-split-file",
        type=Path,
        default=PROJECT_ROOT / "outputs/generated-csd/splits/spider_seed123_train50_test100.json",
    )
    parser.add_argument("--spider-source", choices=["auto", "hf", "local"], default="auto")
    parser.add_argument("--spider-dir", type=Path, default=None)
    parser.add_argument("--spider-train-size", type=int, default=50)
    parser.add_argument("--spider-test-size", type=int, default=100)
    parser.add_argument("--spider-eval-max-steps", type=int, default=400)
    parser.add_argument("--spider-eval-step-token-budget", type=int, default=4)
    parser.add_argument("--spider-vllm-gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--spider-vllm-max-model-len", type=int, default=4096)
    parser.add_argument("--spider-cuda-visible-devices", default=os.environ.get("SPIDER_CUDA_VISIBLE_DEVICES", "2"))
    parser.add_argument("--itergen-repo", type=Path, default=Path("/home/aadivyar/itergen"))
    parser.add_argument("--itergen-device", default="cuda:0")
    parser.add_argument("--itergen-seed", type=int, default=0)
    parser.add_argument("--itergen-recurrence-penalty", type=float, default=0.3)
    parser.add_argument("--itergen-gsm-max-new-tokens", type=int, default=128)
    parser.add_argument("--itergen-smiles-max-new-tokens", type=int, default=512)

    parser.add_argument("--cars-repo", type=Path, default=Path("/home/aadivyar/cars"))
    parser.add_argument("--cars-style", choices=["rs", "ars", "rsft", "cars"], default="cars")
    parser.add_argument("--cars-max-attempts-per-example", type=int, default=2000)
    parser.add_argument("--gsm-cars-max-new-tokens", type=int, default=128)
    parser.add_argument("--spider-cars-max-new-tokens", type=int, default=512)
    parser.add_argument("--cars-cuda-visible-devices", default=os.environ.get("CARS_CUDA_VISIBLE_DEVICES", "1,3"))

    parser.add_argument("--smiles-classes", default="acrylates,chain_extenders,isocyanates")
    parser.add_argument("--smiles-train-samples", type=int, default=50)
    parser.add_argument("--smiles-test-samples", type=int, default=100)
    parser.add_argument("--smiles-max-attempts", type=int, default=2000)
    parser.add_argument("--smiles-eval-max-steps", type=int, default=512)
    parser.add_argument("--smiles-eval-step-token-budget", type=int, default=1)
    parser.add_argument("--smiles-vllm-gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--smiles-vllm-max-model-len", type=int, default=4096)
    parser.add_argument("--smiles-cuda-visible-devices", default=os.environ.get("SMILES_CUDA_VISIBLE_DEVICES", "1,3"))
    args = parser.parse_args()

    args.output_dir = args.output_dir.expanduser().resolve()
    args.gsm_split_file = args.gsm_split_file.expanduser().resolve()
    args.spider_split_file = args.spider_split_file.expanduser().resolve()
    args.itergen_repo = args.itergen_repo.expanduser().resolve()
    args.cars_repo = args.cars_repo.expanduser().resolve()

    models = select_models(args.models)
    datasets = parse_csv(args.datasets, DATASETS, "dataset")
    methods = parse_csv(args.methods, METHODS, "method")
    paths = status_paths(args)
    paths["root"].mkdir(parents=True, exist_ok=True)
    paths["logs"].mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(args, models, datasets, methods)
    write_json(paths["manifest"], manifest)
    write_json(paths["latest"], {
        "run_name": args.run_name,
        "root": str(paths["root"]),
        "ledger": str(paths["ledger"]),
        "manifest": str(paths["manifest"]),
        "started_at": now_iso(),
        "dry_run": args.dry_run,
    })
    append_ledger(paths["ledger"], {"event": "run_start", "status": "running", "manifest": str(paths["manifest"])})

    if "spider" in datasets and any(m in methods for m in ("itergen", "cars", "metadecode")):
        ensure_spider_split(args)

    base_env = os.environ.copy()

    for model in models:
        if "gsm" in datasets and "itergen" in methods:
            cell_id = f"gsm_itergen_{model.alias}"
            output_path = paths["root"] / "benchmarks" / f"{cell_id}.json"
            cmd = gsm_itergen_command(args, model, output_path)
            run_logged(
                args=args,
                ledger=paths["ledger"],
                cell_ids=[cell_id],
                group_id=cell_id,
                cmd=cmd,
                log_path=paths["logs"] / f"{cell_id}.log",
                env=gpu_env(base_env, args.gsm_cuda_visible_devices),
                meta={"dataset": "gsm", "method": "itergen", "model_alias": model.alias, "model_name": model.name, "output_path": str(output_path)},
            )

        if "gsm" in datasets and "cars" in methods:
            cell_id = f"gsm_cars_{model.alias}"
            output_path = paths["root"] / "benchmarks" / f"{cell_id}.json"
            cmd = gsm_cars_command(args, model, output_path)
            run_logged(
                args=args,
                ledger=paths["ledger"],
                cell_ids=[cell_id],
                group_id=cell_id,
                cmd=cmd,
                log_path=paths["logs"] / f"{cell_id}.log",
                env=gpu_env(base_env, args.cars_cuda_visible_devices),
                meta={"dataset": "gsm", "method": "cars", "model_alias": model.alias, "model_name": model.name, "output_path": str(output_path)},
            )

        if "gsm" in datasets and "metadecode" in methods:
            cell_id = f"gsm_metadecode_{model.alias}"
            cmd = gsm_metadecode_command(args, model)
            run_logged(
                args=args,
                ledger=paths["ledger"],
                cell_ids=[cell_id],
                group_id=cell_id,
                cmd=cmd,
                log_path=paths["logs"] / f"{cell_id}.log",
                env=gpu_env(base_env, args.gsm_cuda_visible_devices),
                meta={"dataset": "gsm", "method": "metadecode", "model_alias": model.alias, "model_name": model.name},
            )

        if "spider" in datasets and any(m in methods for m in ("itergen", "metadecode")):
            paired_methods = [m for m in ("itergen", "metadecode") if m in methods]
            cell_ids = [f"spider_{method}_{model.alias}" for method in paired_methods]
            cmd = spider_pair_command(args, model)
            run_logged(
                args=args,
                ledger=paths["ledger"],
                cell_ids=cell_ids,
                group_id=f"spider_itergen_metadecode_{model.alias}",
                cmd=cmd,
                log_path=paths["logs"] / f"spider_itergen_metadecode_{model.alias}.log",
                env=gpu_env(base_env, args.spider_cuda_visible_devices),
                meta={"dataset": "spider", "method": "+".join(paired_methods), "model_alias": model.alias, "model_name": model.name},
            )

        if "spider" in datasets and "cars" in methods:
            cell_id = f"spider_cars_{model.alias}"
            output_path = paths["root"] / "benchmarks" / f"{cell_id}.json"
            cmd = spider_cars_command(args, model, output_path)
            run_logged(
                args=args,
                ledger=paths["ledger"],
                cell_ids=[cell_id],
                group_id=cell_id,
                cmd=cmd,
                log_path=paths["logs"] / f"{cell_id}.log",
                env=gpu_env(base_env, args.cars_cuda_visible_devices),
                meta={"dataset": "spider", "method": "cars", "model_alias": model.alias, "model_name": model.name, "output_path": str(output_path)},
            )

        if "smiles" in datasets and any(m in methods for m in ("cars", "metadecode")):
            paired_methods = [m for m in ("cars", "metadecode") if m in methods]
            cell_ids = [f"smiles_{method}_{model.alias}" for method in paired_methods]
            cmd = smiles_pair_command(args, model)
            run_logged(
                args=args,
                ledger=paths["ledger"],
                cell_ids=cell_ids,
                group_id=f"smiles_cars_metadecode_{model.alias}",
                cmd=cmd,
                log_path=paths["logs"] / f"smiles_cars_metadecode_{model.alias}.log",
                env=gpu_env(base_env, args.smiles_cuda_visible_devices),
                meta={"dataset": "smiles", "method": "+".join(paired_methods), "model_alias": model.alias, "model_name": model.name},
            )

        if "smiles" in datasets and "itergen" in methods:
            cell_id = f"smiles_itergen_{model.alias}"
            output_path = paths["root"] / "benchmarks" / f"{cell_id}.json"
            cmd = smiles_itergen_command(args, model, output_path)
            run_logged(
                args=args,
                ledger=paths["ledger"],
                cell_ids=[cell_id],
                group_id=cell_id,
                cmd=cmd,
                log_path=paths["logs"] / f"{cell_id}.log",
                env=gpu_env(base_env, args.smiles_cuda_visible_devices),
                meta={"dataset": "smiles", "method": "itergen", "model_alias": model.alias, "model_name": model.name, "output_path": str(output_path)},
            )

    for ablation_id, cmd, reason in legacy_ablation_commands(args):
        run_logged(
            args=args,
            ledger=paths["ledger"],
            cell_ids=[ablation_id],
            group_id=ablation_id,
            cmd=cmd,
            log_path=paths["logs"] / f"{ablation_id}.log",
            env=gpu_env(base_env, args.gsm_cuda_visible_devices),
            meta={"dataset": "gsm", "method": "ablation", "model_alias": "legacy_script", "model_name": "legacy_script", "reason": reason},
        )

    append_ledger(paths["ledger"], {"event": "run_end", "status": "completed"})
    latest = json.loads(paths["latest"].read_text())
    latest["completed_at"] = now_iso()
    latest["status"] = "completed"
    write_json(paths["latest"], latest)
    print(f"[summary] manifest={paths['manifest']}")
    print(f"[summary] ledger={paths['ledger']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
