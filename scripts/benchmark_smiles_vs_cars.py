#!/usr/bin/env python3
"""Compare generated CSD strategies against the CARS SMILES benchmark setup."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluations.smiles.dataset import SMILES_CLASSES, get_smiles_task
from evaluations.smiles.metrics import evaluate_smiles_output

MODEL_MAP = {
    "1": "meta-llama/Llama-3.1-8B-Instruct",
    "2": "Qwen/Qwen2.5-7B-Instruct",
    "3": "Qwen/Qwen2.5-14B-Instruct",
}


def _normalize_classes(raw: str | None) -> list[str]:
    if not raw:
        return list(SMILES_CLASSES)
    classes = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = sorted(set(classes) - set(SMILES_CLASSES))
    if unknown:
        raise ValueError(f"Unknown SMILES class(es): {unknown}")
    return classes


def _sha8(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


def _ensure_cars_files(cars_repo: Path, classes: list[str]) -> None:
    missing: list[Path] = []
    for class_name in classes:
        for suffix in ("lark", "txt"):
            path = cars_repo / "datasets" / "smiles" / f"{class_name}.{suffix}"
            if not path.exists():
                missing.append(path)
    if missing:
        raise FileNotFoundError("Missing CARS SMILES files: " + ", ".join(map(str, missing)))
    if not (cars_repo / "run_task.py").exists():
        raise FileNotFoundError(f"Missing CARS run_task.py under {cars_repo}")


def _cars_command(args, class_name: str, log_dir: Path) -> list[str]:
    cars_repo = Path(args.cars_repo).expanduser().resolve()
    grammar = cars_repo / "datasets" / "smiles" / f"{class_name}.lark"
    prompt = cars_repo / "datasets" / "smiles" / f"{class_name}.txt"
    model_name = args.model_name or MODEL_MAP[args.model_number]
    return [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_cars_task.py"),
        "--cars-repo",
        str(cars_repo),
        "--grammar-file",
        str(grammar),
        "--prompt-file",
        str(prompt),
        "--model-name",
        model_name,
        "--sample-style",
        args.cars_style,
        "--log-dir",
        str(log_dir),
        "--target-samples",
        str(args.target_samples),
        "--n-steps",
        str(args.max_attempts),
        "--max-new-tokens",
        str(args.max_steps),
    ]


def _latest_cars_log_dir(cars_repo: Path, class_name: str, style: str, model_number: str) -> Path | None:
    pattern = f"smiles-{class_name}-{class_name}-*-{model_number}/{style}-*"
    candidates = [p for p in (cars_repo / "runs_log").glob(pattern) if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _summarize_cars_log(log_dir: Path, class_name: str, target_samples: int = 50) -> dict[str, Any]:
    task = get_smiles_task(class_name)
    records: list[dict[str, Any]] = []
    success_flags: list[bool] = []
    for json_file in sorted(log_dir.glob("*.json")):
        data = json.loads(json_file.read_text())
        success_flags.extend(bool(x) for x in data.get("successes", []))
        for step in data.get("steps", []):
            output = "".join(step.get("tokens", []))
            eval_row = evaluate_smiles_output(
                class_name,
                output,
                task["grammar_text"],
                task["prompt_exemplars"],
            )
            records.append({
                "output": output,
                "token_count": len(step.get("token_ids", [])),
                "raw_logprob": step.get("raw_logprob"),
                "constrained_logprob": step.get("cons_logprob"),
                **eval_row,
            })
    unique_valid = sorted({r["smiles"] for r in records if r.get("unique_valid_candidate")})
    syntax_count = sum(1 for r in records if r.get("syntax_valid"))
    valid_membership_count = sum(1 for r in records if r.get("valid_class_membership"))
    membership_count_all = sum(1 for r in records if r.get("class_membership"))
    attempts_to_100 = None
    attempts_to_target = None
    successes_seen = 0
    for idx, flag in enumerate(success_flags, start=1):
        if flag:
            successes_seen += 1
        if attempts_to_target is None and successes_seen >= target_samples:
            attempts_to_target = idx
        if successes_seen >= 100:
            attempts_to_100 = idx
    return {
        "class_name": class_name,
        "log_dir": str(log_dir),
        "target_samples": target_samples,
        "attempt_count": len(success_flags),
        "success_count": sum(1 for x in success_flags if x),
        "samples_needed_for_target_successes": attempts_to_target,
        "samples_needed_for_100_successes": attempts_to_100,
        "unique_valid_count": len(unique_valid),
        "syntax_rate": syntax_count / max(1, len(records)),
        "accuracy": valid_membership_count / syntax_count if syntax_count else None,
        "accuracy_definition": "class_membership_among_syntax_valid_molecules",
        "accuracy_num_correct": valid_membership_count,
        "accuracy_denominator": syntax_count,
        "invalid_outputs_excluded_from_accuracy": len(records) - syntax_count,
        "membership_rate_all_attempts": membership_count_all / max(1, len(records)),
        "records": records,
    }


def run_cars(args, classes: list[str]) -> list[dict[str, Any]]:
    cars_repo = Path(args.cars_repo).expanduser().resolve()
    _ensure_cars_files(cars_repo, classes)
    summaries = []
    env = os.environ.copy()
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    for class_name in classes:
        log_dir = args.output_dir / "cars_logs" / class_name
        cmd = _cars_command(args, class_name, log_dir)
        if args.dry_run:
            summaries.append({"class_name": class_name, "command": cmd, "cwd": str(cars_repo)})
            continue
        start = time.time()
        subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)
        if log_dir is None:
            raise RuntimeError(f"CARS completed but no log directory was found for {class_name}")
        summary = _summarize_cars_log(log_dir, class_name, target_samples=args.target_samples)
        summary["wall_time"] = time.time() - start
        summaries.append(summary)
    return summaries


def run_csd(args, classes: list[str]) -> list[dict[str, Any]]:
    spec = importlib.util.spec_from_file_location(
        "csd_synthesis_evaluator", REPO_ROOT / "synthesis" / "evaluator.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load synthesis/evaluator.py")
    evaluator_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = evaluator_module
    spec.loader.exec_module(evaluator_module)
    Evaluator = evaluator_module.Evaluator

    compiled = Path(args.compiled_module).expanduser().resolve()
    model_name = args.model_name or MODEL_MAP[args.model_number]
    summaries: list[dict[str, Any]] = []
    for class_name in classes:
        evaluator = Evaluator(
            dataset_name="smiles",
            model_name=model_name,
            backend=args.backend,
            device=args.device,
            sample_size=1,
            max_steps=args.max_steps,
            step_token_budget=args.step_token_budget,
            smiles_classes=class_name,
        )
        records: list[dict[str, Any]] = []
        unique_valid: set[str] = set()
        start = time.time()
        attempts = 0
        while attempts < args.max_attempts and len(unique_valid) < args.target_samples:
            attempts += 1
            result = evaluator.evaluate_sample(compiled, sample_size=1)
            if not result.success:
                records.append({"error": result.error, "success": False})
                continue
            for sample in result.sample_outputs:
                row = {
                    "output": sample.get("full_output"),
                    "token_count": sample.get("token_count"),
                    "time_seconds": sample.get("time_seconds"),
                    "success": not sample.get("error"),
                    **sample.get("smiles_eval", {}),
                }
                records.append(row)
                if row.get("unique_valid_candidate"):
                    unique_valid.add(row.get("smiles", ""))
        syntax_count = sum(1 for r in records if r.get("syntax_valid"))
        valid_membership_count = sum(1 for r in records if r.get("valid_class_membership"))
        membership_count_all = sum(1 for r in records if r.get("class_membership"))
        summaries.append({
            "class_name": class_name,
            "attempt_count": attempts,
            "success_count": sum(1 for r in records if r.get("success")),
            "unique_valid_count": len(unique_valid),
            "reached_target": len(unique_valid) >= args.target_samples,
            "syntax_rate": syntax_count / max(1, len(records)),
            "accuracy": valid_membership_count / syntax_count if syntax_count else None,
            "accuracy_definition": "class_membership_among_syntax_valid_molecules",
            "accuracy_num_correct": valid_membership_count,
            "accuracy_denominator": syntax_count,
            "invalid_outputs_excluded_from_accuracy": len(records) - syntax_count,
            "membership_rate_all_attempts": membership_count_all / max(1, len(records)),
            "wall_time": time.time() - start,
            "records": records,
        })
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cars-repo", type=str, default=None)
    parser.add_argument("--compiled-module", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/smiles-benchmark"))
    parser.add_argument("--classes", type=str, default=",".join(SMILES_CLASSES))
    parser.add_argument("--model-number", choices=sorted(MODEL_MAP), default="2")
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--backend", choices=["huggingface", "vllm"], default="vllm")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cars-style", choices=["rs", "ars", "rsft", "cars"], default="cars")
    parser.add_argument("--target-samples", type=int, default=50)
    parser.add_argument("--max-attempts", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--step-token-budget", type=int, default=1)
    parser.add_argument("--cuda-visible-devices", type=str, default="1,2")
    parser.add_argument("--run-cars", action="store_true")
    parser.add_argument("--run-csd", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    classes = _normalize_classes(args.classes)
    if args.dry_run and args.cars_repo:
        _ensure_cars_files(Path(args.cars_repo).expanduser().resolve(), classes)

    run_cars_flag = args.run_cars or (not args.run_csd and bool(args.cars_repo))
    run_csd_flag = args.run_csd or (not args.run_cars and bool(args.compiled_module))
    if run_cars_flag and not args.cars_repo:
        raise SystemExit("--run-cars requires --cars-repo")
    if run_csd_flag and not args.compiled_module:
        raise SystemExit("--run-csd requires --compiled-module")

    output = {
        "config": {
            "classes": classes,
            "model_number": args.model_number,
            "model_name": args.model_name or MODEL_MAP[args.model_number],
            "target_samples": args.target_samples,
            "max_attempts": args.max_attempts,
            "dry_run": args.dry_run,
        },
        "cars": run_cars(args, classes) if run_cars_flag else [],
        "csd": run_csd(args, classes) if run_csd_flag and not args.dry_run else [],
    }
    if args.dry_run and run_csd_flag:
        output["csd"] = [{
            "compiled_module": str(Path(args.compiled_module).expanduser()),
            "classes": classes,
            "target_samples": args.target_samples,
            "max_attempts": args.max_attempts,
        }]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"smiles_benchmark_{int(time.time())}.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(json.dumps(output["config"], indent=2))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
