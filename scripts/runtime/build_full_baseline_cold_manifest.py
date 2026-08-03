#!/usr/bin/env python3
"""Build and launch the 2026-08-03 cold campaign after all baselines finish."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from synthesis.evaluate.benchmarks.gsm_symbolic.prompts import GSM_CRANE_COT_TASK


CAMPAIGN = "full-baseline-20260803"
STRATEGIES = ("unconstrained", "gcd", "crane", "itergen", "cars")
EVIDENCE_PATH = Path("saved-results/2026-08-03-full-baseline-campaign-evidence.json")
MANIFEST_PATH = Path("saved-results/2026-08-03-full-baseline-cold-manifest.json")
MODELS = (
    ("qwen25-1p5b", "Qwen/Qwen2.5-1.5B-Instruct", 16_000, 0.30),
    ("qwen25-7b", "Qwen/Qwen2.5-7B-Instruct", 22_000, 0.45),
    ("qwen35-2b", "Qwen/Qwen3.5-2B", 16_384, 0.35),
    ("qwen35-4b", "Qwen/Qwen3.5-4B", 19_000, 0.40),
)
SPIDER_TASK = (
    "Generate a single valid SQL query as exactly `SQL: <<YOUR QUERY>>`, using only "
    "the provided schema context."
)
SMILES_TASK = (
    "Generate valid SMILES strings that match the requested molecular class while "
    "maintaining parser-valid output."
)


@dataclass(frozen=True)
class Cohort:
    slug: str
    dataset: str
    sample_size: int
    heldout_sample_size: int
    max_steps: int
    task: str
    smiles_class: str | None = None


COHORTS = (
    Cohort("gsm", "gsm_symbolic", 49, 49, 900, GSM_CRANE_COT_TASK),
    Cohort("spider", "spider", 300, 300, 176, SPIDER_TASK),
    Cohort("smiles-acrylates", "smiles", 50, 100, 400, SMILES_TASK, "acrylates"),
    Cohort(
        "smiles-chain_extenders",
        "smiles",
        50,
        100,
        400,
        SMILES_TASK,
        "chain_extenders",
    ),
    Cohort(
        "smiles-isocyanates",
        "smiles",
        50,
        100,
        400,
        SMILES_TASK,
        "isocyanates",
    ),
)


class CampaignError(ValueError):
    pass


def baseline_artifact(
    repo: Path, cohort: Cohort, model: tuple[str, str, int, float], strategy: str
) -> Path:
    return (
        repo
        / "outputs/baselines/full_baseline_20260803"
        / cohort.slug
        / model[0]
        / f"{strategy}.json"
    )


def _rate_count(value: Any, total: int, label: str, path: Path) -> int:
    if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
        raise CampaignError(f"{path}: invalid {label}")
    count = round(float(value) * total)
    if abs(float(value) - count / total) > 1e-9:
        raise CampaignError(f"{path}: {label} does not resolve to an exact count")
    return count


def _read_baseline(path: Path, total: int, strategy: str, repo: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CampaignError(f"missing baseline artifact: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignError(f"invalid baseline artifact {path}: {exc}") from exc
    answers = payload.get("answers")
    if not isinstance(answers, list) or len(answers) != total:
        raise CampaignError(f"{path}: answer count must be {total}")
    if any(not isinstance(row, dict) or "generated_answer" not in row for row in answers):
        raise CampaignError(f"{path}: every row must contain generated_answer")
    metrics = payload.get("metrics") or {}
    if int(metrics.get("num_examples") or -1) != total:
        raise CampaignError(f"{path}: metrics.num_examples must be {total}")
    relative = path.relative_to(repo)
    return {
        "strategy": strategy,
        "num_correct": _rate_count(payload.get("accuracy"), total, "accuracy", path),
        "syntax_count": _rate_count(payload.get("syntax_rate"), total, "syntax_rate", path),
        "num_examples": total,
        "accuracy": float(payload["accuracy"]),
        "syntax_rate": float(payload["syntax_rate"]),
        "source_artifact": str(relative),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _cell_id(cohort: Cohort, model_slug: str) -> str:
    return f"{cohort.slug}-{model_slug}"


def build_campaign(repo: Path, git_commit: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(git_commit) != 40:
        raise CampaignError("git_commit must be a full 40-character hash")
    evidence_cells: dict[str, Any] = {}
    jobs: list[dict[str, Any]] = []
    for cohort in COHORTS:
        for model in MODELS:
            model_slug, eval_model, reservation_mib, gpu_util = model
            cell = _cell_id(cohort, model_slug)
            rows = [
                _read_baseline(
                    baseline_artifact(repo, cohort, model, strategy),
                    cohort.sample_size,
                    strategy,
                    repo,
                )
                for strategy in STRATEGIES
            ]
            max_correct = max(row["num_correct"] for row in rows)
            max_syntax = max(row["syntax_count"] for row in rows)
            if max_correct == cohort.sample_size:
                min_accuracy = 0.95
                threshold_policy = "perfect_baseline_95_percent_exception"
            else:
                min_accuracy = (max_correct + 1) / cohort.sample_size
                threshold_policy = "strict_plus_one"
            min_syntax = min(max_syntax / cohort.sample_size, 0.90)
            evidence_cells[cell] = {
                "dataset": cohort.dataset,
                "eval_model": eval_model,
                "split_name": "train",
                "smiles_class": cohort.smiles_class,
                "num_examples": cohort.sample_size,
                "baselines": rows,
                "max_accuracy_count": max_correct,
                "max_accuracy_strategies": [
                    row["strategy"] for row in rows if row["num_correct"] == max_correct
                ],
                "max_syntax_count": max_syntax,
                "max_syntax_strategies": [
                    row["strategy"] for row in rows if row["syntax_count"] == max_syntax
                ],
                "min_accuracy": min_accuracy,
                "min_syntax_rate": min_syntax,
                "threshold_policy": threshold_policy,
            }
            output_name = f"coldq_fullbaseline_20260803_{cell}"
            job: dict[str, Any] = {
                "cell_id": cell,
                "task": cohort.task,
                "dataset": cohort.dataset,
                "eval_model": eval_model,
                "max_iterations": 40,
                "interrupted_author_calls": 0,
                "eval_sample_size": cohort.sample_size,
                "baseline_num_correct": max_correct,
                "baseline_num_examples": cohort.sample_size,
                "baseline_source": str(EVIDENCE_PATH),
                "min_accuracy": min_accuracy,
                "min_syntax_rate": min_syntax,
                "threshold_policy": threshold_policy,
                "eval_max_steps": cohort.max_steps,
                "eval_max_seconds": 90.0,
                "memory_reservation_mib": reservation_mib,
                "gpu_mem_util": gpu_util,
                "output_name": output_name,
                "log_file": f"outputs/generated/{output_name}/run.log",
                "heldout_sample_size": cohort.heldout_sample_size,
                "heldout_split_name": "test",
                "heldout_output_json": str(
                    repo / "outputs/reeval/full_baseline_20260803" / f"{cell}.json"
                ),
                "claude_config_dir": "/home/aadivyar/.claude-csd-synthesis",
                "claude_expected_account": "aadivya@fermi.ai",
            }
            if cohort.dataset == "gsm_symbolic":
                job["heldout_split_file"] = (
                    "environment/benchmark_splits/"
                    "gsm_symbolic_crane_proportional_49x49_seed123.json"
                )
            elif cohort.dataset == "spider":
                job["heldout_split_file"] = (
                    "environment/benchmark_splits/"
                    "spider_dev_proportional_300x300_seed334.json"
                )
            else:
                job["smiles_class"] = cohort.smiles_class
            jobs.append(job)
    evidence = {
        "campaign": CAMPAIGN,
        "git_commit": git_commit,
        "baseline_strategies": list(STRATEGIES),
        "cells": evidence_cells,
    }
    manifest = {
        "campaign": CAMPAIGN,
        "git_commit": git_commit,
        "approved_author_call_cap": 800,
        "jobs": jobs,
    }
    return evidence, manifest


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_campaign(jobs: list[dict[str, Any]], repo: Path) -> None:
    expected_cells = {
        _cell_id(cohort, model[0]) for cohort in COHORTS for model in MODELS
    }
    if len(jobs) != 20 or {str(job.get("cell_id")) for job in jobs} != expected_cells:
        raise CampaignError("full baseline cold campaign must contain exactly 20 cells")
    if sum(int(job.get("max_iterations") or 0) for job in jobs) != 800:
        raise CampaignError("full baseline cold campaign must contain exactly 800 attempts")
    evidence_path = repo / EVIDENCE_PATH
    try:
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignError(f"invalid campaign evidence: {exc}") from exc
    if evidence.get("campaign") != CAMPAIGN:
        raise CampaignError("campaign evidence has the wrong campaign id")
    for job in jobs:
        cell = str(job["cell_id"])
        entry = (evidence.get("cells") or {}).get(cell) or {}
        rows = entry.get("baselines") or []
        if [row.get("strategy") for row in rows] != list(STRATEGIES):
            raise CampaignError(f"{cell}: evidence must include all five baselines")
        for row in rows:
            source = repo / str(row.get("source_artifact", ""))
            if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != row.get(
                "source_sha256"
            ):
                raise CampaignError(f"{cell}: baseline artifact hash mismatch")
        total = int(job["eval_sample_size"])
        max_correct = max(int(row["num_correct"]) for row in rows)
        max_syntax = max(int(row["syntax_count"]) for row in rows)
        if max_correct == total:
            expected_accuracy = 0.95
            expected_policy = "perfect_baseline_95_percent_exception"
        else:
            expected_accuracy = (max_correct + 1) / total
            expected_policy = "strict_plus_one"
        expected_syntax = min(max_syntax / total, 0.90)
        if (
            int(job["baseline_num_correct"]) != max_correct
            or int(job["baseline_num_examples"]) != total
            or abs(float(job["min_accuracy"]) - expected_accuracy) > 1e-12
            or abs(float(job["min_syntax_rate"]) - expected_syntax) > 1e-12
            or job.get("threshold_policy") != expected_policy
        ):
            raise CampaignError(f"{cell}: thresholds do not match baseline maxima")
        if job.get("claude_expected_account") != "aadivya@fermi.ai":
            raise CampaignError(f"{cell}: wrong approved Claude account")
        if job.get("claude_config_dir") != "/home/aadivyar/.claude-csd-synthesis":
            raise CampaignError(f"{cell}: wrong Claude config directory")


def controller_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--baseline-controller-pid", type=int, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--gpus", default="0,2,3")
    args = parser.parse_args()
    while True:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=args.repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        try:
            evidence, manifest = build_campaign(args.repo, commit)
            break
        except CampaignError as exc:
            if not controller_is_alive(args.baseline_controller_pid):
                raise CampaignError(
                    f"baseline controller exited before the matrix was complete: {exc}"
                ) from exc
            print(f"[full-baseline-cold] waiting: {exc}", flush=True)
            time.sleep(args.poll_seconds)
    _atomic_json(args.repo / EVIDENCE_PATH, evidence)
    _atomic_json(args.repo / MANIFEST_PATH, manifest)
    validate_campaign(manifest["jobs"], args.repo)
    command = [
        str(args.python),
        "-m",
        "scripts.runtime.run_cold_synthesis_queue",
        "--repo",
        str(args.repo),
        "--manifest",
        str(args.repo / MANIFEST_PATH),
        "--python",
        str(args.python),
        "--lock-file",
        str(args.repo / ".context/full_baseline_20260803_cold.lock"),
        "--state-dir",
        str(args.repo / ".context/full_baseline_20260803_cold_state"),
        "--campaign-profile",
        CAMPAIGN,
        "--gpus",
        args.gpus,
        "--poll-seconds",
        str(args.poll_seconds),
    ]
    print(f"[full-baseline-cold] launching {len(manifest['jobs'])} jobs", flush=True)
    return subprocess.run(command, cwd=args.repo).returncode


if __name__ == "__main__":
    raise SystemExit(main())
