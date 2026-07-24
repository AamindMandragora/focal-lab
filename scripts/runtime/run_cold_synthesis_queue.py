#!/usr/bin/env python3
"""Run the approved cold-only synthesis campaign across available focal GPUs."""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from synthesis.evaluate.benchmarks.gsm_symbolic.prompts import GSM_CRANE_COT_TASK
from synthesis.run_constants import VLLM_GPU_MEMORY_UTILIZATION
from scripts.runtime.run_warm_task_recovery_queue import (
    GPU_SAFETY_MIB,
    gpu_memory_snapshot,
    required_memory_mib,
)


AUTHOR_MODEL = "claude-sonnet-4-6"
TERMINAL_SYNTHESIS_FAILURE = 75
POOLABLE_DATASETS = {"gsm_symbolic", "spider"}
POOLABLE_GPU_COUNT = 3


def _human_log_emit(repo: Path, cell_id: str, marker: str, detail: str = "") -> None:
    """Best-effort write to master + gsm/spider/smiles lane logs.

    Callers: run_job COLDQ start/finish markers.
    User instruction: human-logs (master + gsm/spider/smiles lane logs).
    """
    try:
        from scripts.runtime.zero_acc_babysitter.human_logs import HumanLogWriter
    except ImportError:
        return
    HumanLogWriter(repo).emit(cell_id, marker, detail)


BASELINE_STRATEGY_BY_DATASET = {
    "gsm_symbolic": "crane",
    "spider": "itergen",
    "smiles": "cars",
}
CANONICAL_TRAIN_SPLIT_BY_DATASET = {
    "gsm_symbolic": Path(
        "environment/benchmark_splits/gsm_symbolic_crane_proportional_49x49_seed123.json"
    ),
    "spider": Path(
        "environment/benchmark_splits/spider_dev_proportional_300x300_seed334.json"
    ),
}
PINNED_CODE_PATHS = (
    "synthesis",
    "scripts",
    "environment/benchmark_splits",
    "legacy/CRANE/src/gsm_symbolic",
)
GSM_TASK = GSM_CRANE_COT_TASK
SPIDER_TASK = (
    "Generate a single valid SQL query as exactly `SQL: <<YOUR QUERY>>`, using only the "
    "provided schema context."
)
SMILES_TASK = (
    "Generate valid SMILES strings that match the requested molecular class while "
    "maintaining parser-valid output."
)
APPROVED_AUTHOR_CALL_CAP = 802
EXPECTED_CELLS: dict[str, dict[str, Any]] = {
    "gsm-qwen25-1p5b": {"dataset": "gsm_symbolic", "eval_model": "Qwen/Qwen2.5-1.5B-Instruct", "max_iterations": 38, "interrupted_author_calls": 3, "eval_sample_size": 49, "heldout_sample_size": 49, "eval_max_steps": 900, "task": GSM_TASK},
    "gsm-qwen25-7b": {"dataset": "gsm_symbolic", "eval_model": "Qwen/Qwen2.5-7B-Instruct", "max_iterations": 38, "interrupted_author_calls": 3, "eval_sample_size": 49, "heldout_sample_size": 49, "eval_max_steps": 900, "task": GSM_TASK},
    "gsm-qwen35-2b": {"dataset": "gsm_symbolic", "eval_model": "Qwen/Qwen3.5-2B", "max_iterations": 38, "interrupted_author_calls": 2, "eval_sample_size": 49, "heldout_sample_size": 49, "eval_max_steps": 900, "task": GSM_TASK},
    "gsm-qwen35-4b": {"dataset": "gsm_symbolic", "eval_model": "Qwen/Qwen3.5-4B", "max_iterations": 40, "interrupted_author_calls": 0, "eval_sample_size": 49, "heldout_sample_size": 49, "eval_max_steps": 900, "task": GSM_TASK},
    "spider-qwen25-1p5b": {"dataset": "spider", "eval_model": "Qwen/Qwen2.5-1.5B-Instruct", "max_iterations": 40, "interrupted_author_calls": 0, "eval_sample_size": 300, "heldout_sample_size": 300, "eval_max_steps": 200, "task": SPIDER_TASK},
    "spider-qwen25-7b": {"dataset": "spider", "eval_model": "Qwen/Qwen2.5-7B-Instruct", "max_iterations": 40, "interrupted_author_calls": 0, "eval_sample_size": 300, "heldout_sample_size": 300, "eval_max_steps": 200, "task": SPIDER_TASK},
    "spider-qwen35-2b": {"dataset": "spider", "eval_model": "Qwen/Qwen3.5-2B", "max_iterations": 40, "interrupted_author_calls": 0, "eval_sample_size": 300, "heldout_sample_size": 300, "eval_max_steps": 200, "task": SPIDER_TASK},
    "spider-qwen35-4b": {"dataset": "spider", "eval_model": "Qwen/Qwen3.5-4B", "max_iterations": 40, "interrupted_author_calls": 0, "eval_sample_size": 300, "heldout_sample_size": 300, "eval_max_steps": 200, "task": SPIDER_TASK},
    "smiles-acrylates-qwen25-1p5b": {"dataset": "smiles", "eval_model": "Qwen/Qwen2.5-1.5B-Instruct", "max_iterations": 40, "interrupted_author_calls": 0, "eval_sample_size": 50, "heldout_sample_size": 100, "eval_max_steps": 400, "task": SMILES_TASK, "smiles_class": "acrylates"},
    "smiles-acrylates-qwen25-7b": {"dataset": "smiles", "eval_model": "Qwen/Qwen2.5-7B-Instruct", "max_iterations": 40, "interrupted_author_calls": 0, "eval_sample_size": 50, "heldout_sample_size": 100, "eval_max_steps": 400, "task": SMILES_TASK, "smiles_class": "acrylates"},
    "smiles-acrylates-qwen35-2b": {"dataset": "smiles", "eval_model": "Qwen/Qwen3.5-2B", "max_iterations": 40, "interrupted_author_calls": 0, "eval_sample_size": 50, "heldout_sample_size": 100, "eval_max_steps": 400, "task": SMILES_TASK, "smiles_class": "acrylates"},
    "smiles-acrylates-qwen35-4b": {"dataset": "smiles", "eval_model": "Qwen/Qwen3.5-4B", "max_iterations": 40, "interrupted_author_calls": 0, "eval_sample_size": 50, "heldout_sample_size": 100, "eval_max_steps": 400, "task": SMILES_TASK, "smiles_class": "acrylates"},
    "smiles-chain_extenders-qwen25-1p5b": {"dataset": "smiles", "eval_model": "Qwen/Qwen2.5-1.5B-Instruct", "max_iterations": 40, "interrupted_author_calls": 0, "eval_sample_size": 50, "heldout_sample_size": 100, "eval_max_steps": 400, "task": SMILES_TASK, "smiles_class": "chain_extenders"},
    "smiles-chain_extenders-qwen25-7b": {"dataset": "smiles", "eval_model": "Qwen/Qwen2.5-7B-Instruct", "max_iterations": 40, "interrupted_author_calls": 0, "eval_sample_size": 50, "heldout_sample_size": 100, "eval_max_steps": 400, "task": SMILES_TASK, "smiles_class": "chain_extenders"},
    "smiles-chain_extenders-qwen35-2b": {"dataset": "smiles", "eval_model": "Qwen/Qwen3.5-2B", "max_iterations": 40, "interrupted_author_calls": 0, "eval_sample_size": 50, "heldout_sample_size": 100, "eval_max_steps": 400, "task": SMILES_TASK, "smiles_class": "chain_extenders"},
    "smiles-chain_extenders-qwen35-4b": {"dataset": "smiles", "eval_model": "Qwen/Qwen3.5-4B", "max_iterations": 40, "interrupted_author_calls": 0, "eval_sample_size": 50, "heldout_sample_size": 100, "eval_max_steps": 400, "task": SMILES_TASK, "smiles_class": "chain_extenders"},
    "smiles-isocyanates-qwen25-1p5b": {"dataset": "smiles", "eval_model": "Qwen/Qwen2.5-1.5B-Instruct", "max_iterations": 40, "interrupted_author_calls": 0, "eval_sample_size": 50, "heldout_sample_size": 100, "eval_max_steps": 400, "task": SMILES_TASK, "smiles_class": "isocyanates"},
    "smiles-isocyanates-qwen25-7b": {"dataset": "smiles", "eval_model": "Qwen/Qwen2.5-7B-Instruct", "max_iterations": 40, "interrupted_author_calls": 0, "eval_sample_size": 50, "heldout_sample_size": 100, "eval_max_steps": 400, "task": SMILES_TASK, "smiles_class": "isocyanates"},
    "smiles-isocyanates-qwen35-2b": {"dataset": "smiles", "eval_model": "Qwen/Qwen3.5-2B", "max_iterations": 40, "interrupted_author_calls": 0, "eval_sample_size": 50, "heldout_sample_size": 100, "eval_max_steps": 400, "task": SMILES_TASK, "smiles_class": "isocyanates"},
    "smiles-isocyanates-qwen35-4b": {"dataset": "smiles", "eval_model": "Qwen/Qwen3.5-4B", "max_iterations": 40, "interrupted_author_calls": 0, "eval_sample_size": 50, "heldout_sample_size": 100, "eval_max_steps": 400, "task": SMILES_TASK, "smiles_class": "isocyanates"},
}
EXPECTED_RUNTIME_BY_MODEL = {
    "Qwen/Qwen2.5-1.5B-Instruct": {"memory_reservation_mib": 16000, "gpu_mem_util": 0.4},
    "Qwen/Qwen2.5-7B-Instruct": {"memory_reservation_mib": 22000, "gpu_mem_util": 0.45},
    "Qwen/Qwen2.5-14B-Instruct": {"memory_reservation_mib": 34000, "gpu_mem_util": 0.81},
    "Qwen/Qwen3.5-2B": {"memory_reservation_mib": 16384, "gpu_mem_util": 0.4},
    "Qwen/Qwen3.5-4B": {"memory_reservation_mib": 19000, "gpu_mem_util": 0.45},
    "Qwen/Qwen3.5-9B": {"memory_reservation_mib": 25000, "gpu_mem_util": 0.6},
}
logger = logging.getLogger("cold-synthesis-queue")


class _CombinedLogWriter:
    """Duplicate writes to per-cell and shared campaign logs.

    Do not implement fileno(): subprocess would otherwise dup2 only the first
    stream and skip the combined campaign log.
    """

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def _run_command_teeing_stdout(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None,
    streams: tuple[Any, ...],
) -> int:
    """Run command and copy each stdout line to every stream (line-buffered)."""
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        for stream in streams:
            stream.write(line)
            stream.flush()
    return int(process.wait())


class ConfigError(ValueError):
    pass


def _is_pinned_code_path(path: str) -> bool:
    return any(path == root or path.startswith(f"{root}/") for root in PINNED_CODE_PATHS)


def _dirty_code_files(repo: Path) -> tuple[set[str], set[str]]:
    tracked: set[str] = set()
    for extra in ([], ["--cached"]):
        output = subprocess.run(
            ["git", "diff", *extra, "--name-only", "--", *PINNED_CODE_PATHS],
            cwd=repo,
            text=True,
            capture_output=True,
            check=True,
        ).stdout
        tracked.update(line for line in output.splitlines() if line)
    untracked_output = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "--", *PINNED_CODE_PATHS],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    return tracked, {line for line in untracked_output.splitlines() if line}


def _verify_repair_attestation(
    repo: Path,
    pinned_commit: str,
    dirty_files: set[str],
    attestation_path: Path | None,
) -> None:
    if attestation_path is None or not attestation_path.is_file():
        raise ConfigError("repair attestation is missing for dirty pinned code")
    try:
        payload = json.loads(attestation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"repair attestation is invalid: {exc}") from exc
    files = payload.get("files")
    if payload.get("base_commit") != pinned_commit or payload.get("verifier_exit") != 0:
        raise ConfigError("repair attestation does not match the pinned commit and verifier")
    if not isinstance(files, dict):
        raise ConfigError("repair attestation files must be an object")
    pinned_attested = {str(path): str(digest) for path, digest in files.items() if _is_pinned_code_path(str(path))}
    if set(pinned_attested) != dirty_files:
        raise ConfigError("repair attestation file set does not match dirty pinned code")
    for relative, expected_hash in pinned_attested.items():
        path = repo / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise ConfigError(f"repair attestation hash mismatch for {relative}")


def verify_repo_version(
    repo: Path, pinned_commit: str, repair_attestation: Path | None = None
) -> str:
    live_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True, capture_output=True, check=True
    ).stdout.strip()
    if live_commit != pinned_commit:
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", pinned_commit, live_commit], cwd=repo
        )
        if ancestor.returncode != 0:
            raise ConfigError(
                f"manifest commit {pinned_commit} is not an ancestor of repo commit {live_commit}"
            )
        code_drift = subprocess.run(
            ["git", "diff", "--quiet", f"{pinned_commit}..{live_commit}", "--", *PINNED_CODE_PATHS],
            cwd=repo,
        )
        if code_drift.returncode != 0:
            raise ConfigError(f"code changed after pinned manifest commit {pinned_commit}")
    tracked, untracked = _dirty_code_files(repo)
    if tracked or untracked:
        if repair_attestation is None:
            if tracked:
                raise ConfigError("uncommitted code changes exist in a pinned code path")
            raise ConfigError(
                "untracked code changes exist in a pinned code path: " + ", ".join(sorted(untracked))
            )
        _verify_repair_attestation(repo, pinned_commit, tracked | untracked, repair_attestation)
    return live_commit


def synthesis_command(job: dict[str, Any], python: Path) -> list[str]:
    command = [
        str(python),
        "-m",
        "synthesis.run_synthesis",
        "--task",
        str(job["task"]),
        "--dataset",
        str(job["dataset"]),
        "--min-accuracy",
        str(job["min_accuracy"]),
        "--min-syntax-rate",
        str(job["min_syntax_rate"]),
        "--max-iterations",
        str(job["max_iterations"]),
        "--eval-model",
        str(job["eval_model"]),
        "--eval-sample-size",
        str(job["eval_sample_size"]),
        "--eval-max-steps",
        str(job["eval_max_steps"]),
        "--eval-step-token-budget",
        "1",
        "--eval-max-seconds-per-example",
        str(job["eval_max_seconds"]),
        "--generation-model",
        AUTHOR_MODEL,
        "--generation-backend",
        "claude",
        "--synthesis-max-tokens",
        "8192",
        "--device",
        "auto",
    ]
    if job["dataset"] == "smiles":
        command.extend(
            [
                "--smiles-classes",
                str(job["smiles_class"]),
                "--smiles-samples-per-class",
                str(job["eval_sample_size"]),
                "--smiles-final-samples-per-class",
                str(job["heldout_sample_size"]),
            ]
        )
    return command


def synthesis_environment(
    job: dict[str, Any], gpus: tuple[int, ...], inherited: dict[str, str], repo: Path
) -> dict[str, str]:
    expected = required_gpu_count(job)
    if len(gpus) != expected or len(set(gpus)) != len(gpus):
        raise ConfigError(
            f"{job['cell_id']} requires {expected} distinct GPU(s), got {gpus}"
        )
    gpu_list = ",".join(str(gpu) for gpu in gpus)
    env = dict(inherited)
    env.pop("CSD_EVAL_POOL_SIZE", None)
    env.pop("CSD_EVAL_GPU_SLOTS", None)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": gpu_list,
            "CSD_OUTPUT_NAME": str(job["output_name"]),
            "CSD_OUTPUT_DIR": str(repo / "outputs" / "generated" / str(job["output_name"])),
            "PYTHONUNBUFFERED": "1",
            # Claude Code Max author (matches incident-monitor account isolation)
            "CSD_CLAUDE_EXECUTABLE": "/home/aadivyar/.local/bin/claude",
            "CSD_CLAUDE_CONFIG_DIR": "/home/aadivyar/.claude-csd-synthesis",
            "CSD_CLAUDE_EXPECTED_ACCOUNT": "aadivya@fermi.ai",
            # Per-job vLLM budget; without this run_synthesis falls back to the
            # global 0.81 and pooled eval workers OOM on shared GPUs.
            "CSD_VLLM_GPU_MEMORY_UTILIZATION": str(job["gpu_mem_util"]),
        }
    )
    if job["dataset"] in POOLABLE_DATASETS:
        env["CSD_EVAL_GPU_SLOTS"] = gpu_list
    if job["dataset"] == "smiles":
        env["CSD_CONSTRAINED_TEMPERATURE"] = "0.7"
    return env


def author_free_environment(
    inherited: dict[str, str], gpu: int, *, dataset: str | None = None
) -> dict[str, str]:
    clean = {
        key: value
        for key, value in inherited.items()
        if not key.startswith(("AWS_", "BEDROCK_")) and not key.endswith("_API_KEY")
    }
    clean.pop("CSD_EVAL_GPU_SLOTS", None)
    clean.pop("CSD_EVAL_POOL_SIZE", None)
    clean["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if dataset == "smiles":
        clean["CSD_CONSTRAINED_TEMPERATURE"] = "0.7"
    return clean


def synthesis_required_memory_mib(job: dict[str, Any], gpu_total_mib: int) -> int:
    scheduling_job = {
        **job,
        "gpu_mem_util": VLLM_GPU_MEMORY_UTILIZATION,
    }
    return required_memory_mib(scheduling_job, gpu_total_mib)


def required_gpu_count(job: dict[str, Any]) -> int:
    """Use two workers for stateless GSM/Spider and one for stateful SMILES."""
    return POOLABLE_GPU_COUNT if job["dataset"] in POOLABLE_DATASETS else 1


def choose_gpu_bundle(
    job: dict[str, Any],
    snapshots: dict[int, dict[str, int]],
    reservations: dict[int, dict[str, int]],
    baseline_snapshots: dict[int, dict[str, int]],
) -> tuple[int, ...] | None:
    """Return the least-used disjoint physical GPU bundle that fits one cell."""
    # Callers: cold-queue controller dispatch loop in this module (~L1100+).
    # Existing file; purpose unchanged. No data-file schema change.
    # User: "what's causing this error on focal for data collection rn? pls fix"
    candidates: list[tuple[int, int]] = []
    for gpu, snapshot in snapshots.items():
        required = synthesis_required_memory_mib(job, snapshot["total_mib"])
        reserved = sum(reservations.get(gpu, {}).values())
        baseline_used = baseline_snapshots.get(gpu, snapshot)["used_mib"]
        projected_used = max(snapshot["used_mib"], baseline_used + reserved)
        # Match vLLM: free must be >= util * total (see EngineCore request_memory).
        free_mib = int(
            snapshot.get("free_mib", snapshot["total_mib"] - snapshot["used_mib"])
        )
        projected_free = snapshot["total_mib"] - projected_used
        if free_mib < required + GPU_SAFETY_MIB:
            continue
        if projected_free < required + GPU_SAFETY_MIB:
            continue
        if projected_used + required <= snapshot["total_mib"] - GPU_SAFETY_MIB:
            candidates.append((projected_used, gpu))
    needed = required_gpu_count(job)
    if len(candidates) < needed:
        return None
    return tuple(gpu for _, gpu in sorted(candidates)[:needed])


def current_run_dir(repo: Path, output_name: str) -> Path | None:
    output_root = repo / "outputs" / "generated" / output_name
    latest_file = output_root / "latest_run.txt"
    latest_link = output_root / "latest"
    if latest_file.is_file():
        run_dir = Path(latest_file.read_text(encoding="utf-8").strip())
    elif latest_link.exists():
        run_dir = latest_link.resolve()
    else:
        return None
    return run_dir if run_dir.is_absolute() else repo / run_dir


def report_matches_job(
    report: dict[str, Any], job: dict[str, Any], *, require_exhausted: bool
) -> bool:
    config = report.get("run_configuration") or {}
    thresholds = config.get("thresholds") or {}
    author = config.get("author_model") or {}
    evaluation = config.get("evaluation") or {}
    try:
        total_attempts = int(report.get("total_attempts"))
        max_iterations = int(job["max_iterations"])
    except (TypeError, ValueError, KeyError):
        return False
    if total_attempts < 1 or total_attempts > max_iterations:
        return False
    if require_exhausted and total_attempts != max_iterations:
        return False
    try:
        exact = (
            config.get("task_description") == job["task"]
            and config.get("output_name") == job["output_name"]
            and config.get("git_commit") == job.get("launch_commit", job["git_commit"])
            and int(config.get("max_iterations") or -1) == max_iterations
            and author.get("backend") == "claude"
            and author.get("model") == AUTHOR_MODEL
            and int(author.get("max_new_tokens") or -1) == 8192
            and int(author.get("reasoning_budget_tokens") or -1) == 4096
            and author.get("anthropic_thinking") == "always-on"
            and author.get("anthropic_effort") == "high"
            and author.get("anthropic_thinking_display") == "summarized"
            and evaluation.get("dataset") == job["dataset"]
            and evaluation.get("eval_model") == job["eval_model"]
            and int(evaluation.get("eval_sample_size") or -1)
            == int(job["eval_sample_size"])
            and int(evaluation.get("eval_max_steps") or -1)
            == int(job["eval_max_steps"])
            and int(evaluation.get("eval_step_token_budget") or -1) == 1
            and float(evaluation.get("eval_max_seconds_per_example") or -1)
            == float(job["eval_max_seconds"])
            and abs(
                float(thresholds.get("min_accuracy") or 0)
                - float(job["min_accuracy"])
            )
            <= 1e-12
            and abs(
                float(thresholds.get("min_syntax_rate") or 0)
                - float(job["min_syntax_rate"])
            )
            <= 1e-12
        )
    except (TypeError, ValueError, KeyError):
        return False
    if not exact:
        return False
    dataset = str(job["dataset"])
    expected_smiles_classes = (
        [str(job["smiles_class"])] if dataset == "smiles" else None
    )
    if evaluation.get("smiles_classes") != expected_smiles_classes:
        return False
    if dataset in {"gsm_symbolic", "spider"}:
        split = evaluation.get("split_provenance") or {}
        if split.get("bar_split_name") != "train":
            return False
        prefix = "gsm" if dataset == "gsm_symbolic" else "spider"
        expected_file = (
            "gsm_symbolic_crane_proportional_49x49_seed123.json"
            if prefix == "gsm"
            else "spider_dev_proportional_300x300_seed334.json"
        )
        if split.get(f"{prefix}_split_name") != "train":
            return False
        if Path(str(split.get(f"{prefix}_split_file"))).name != expected_file:
            return False
    return True


def compiled_csd(
    repo: Path,
    output_name: str,
    *,
    min_accuracy: float,
    min_syntax_rate: float,
    job: dict[str, Any],
) -> Path | None:
    run_dir = current_run_dir(repo, output_name)
    if run_dir is None:
        return None
    success_report = run_dir / "results" / "success_report.json"
    if success_report.is_file():
        try:
            payload = json.loads(success_report.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "[coldq] incomplete success report path=%s error=%s",
                success_report,
                exc,
            )
            return None
        if not report_matches_job(payload, job, require_exhausted=False):
            return None
        compiled_dir = Path(str(payload.get("compiled_dir", "")))
    else:
        failure_report = run_dir / "results" / "failure_report.json"
        if not failure_report.is_file():
            return None
        try:
            payload = json.loads(failure_report.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "[coldq] incomplete failure report path=%s error=%s",
                failure_report,
                exc,
            )
            return None
        if not report_matches_job(payload, job, require_exhausted=True):
            return None
        candidates = []
        for attempt in payload.get("attempts", []):
            compilation = attempt.get("compilation") or {}
            evaluation = attempt.get("evaluation") or {}
            if not compilation.get("success") or not compilation.get("output_dir"):
                continue
            if int(evaluation.get("num_examples") or 0) < 1:
                continue
            accuracy = float(evaluation.get("accuracy") or 0.0)
            syntax_rate = float(evaluation.get("syntax_rate") or 0.0)
            shortfall = max(0.0, min_accuracy - accuracy) + max(
                0.0, min_syntax_rate - syntax_rate
            )
            candidates.append(
                (
                    shortfall,
                    -accuracy,
                    -syntax_rate,
                    int(attempt.get("attempt_number") or 0),
                    Path(str(compilation["output_dir"])),
                )
            )
        if not candidates:
            return None
        compiled_dir = min(candidates, key=lambda candidate: candidate[:4])[-1]
    if not compiled_dir.is_absolute():
        compiled_dir = repo / compiled_dir
    candidate = compiled_dir / "GeneratedCSD.py"
    return candidate if candidate.is_file() else None


def synthesis_was_exhausted(
    repo: Path,
    output_name: str,
    job: dict[str, Any],
    *,
    previous_run_dir: Path | None = None,
) -> bool:
    run_dir = current_run_dir(repo, output_name)
    if run_dir is None or run_dir == previous_run_dir:
        return False
    results_dir = run_dir / "results"
    failure_report = results_dir / "failure_report.json"
    if not failure_report.is_file() or (results_dir / "success_report.json").is_file():
        return False
    try:
        payload = json.loads(failure_report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return report_matches_job(payload, job, require_exhausted=True)


def heldout_is_complete(path: Path, job: dict[str, Any]) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    expected = int(job["heldout_sample_size"])
    metrics = payload.get("metrics") or {}
    if int(metrics.get("num_examples") or 0) != expected:
        return False
    answers = payload.get("answers")
    if not isinstance(answers, list) or len(answers) != expected:
        return False
    if not isinstance(payload.get("accuracy"), (int, float)) or not isinstance(
        payload.get("syntax_rate"), (int, float)
    ):
        return False
    provenance = payload.get("reevaluation_provenance") or {}
    compiled_path = Path(str(provenance.get("compiled_csd_path", "")))
    if not compiled_path.is_file():
        return False
    try:
        compiled_hash = hashlib.sha256(compiled_path.read_bytes()).hexdigest()
    except OSError:
        return False
    try:
        provenance_matches = (
            provenance.get("cell_id") == job["cell_id"]
            and provenance.get("manifest_commit") == job["git_commit"]
            and provenance.get("dataset") == job["dataset"]
            and provenance.get("eval_model") == job["eval_model"]
            and provenance.get("compiled_csd_sha256") == compiled_hash
            and int(provenance.get("sample_size") or -1) == expected
            and int(provenance.get("max_steps") or -1) == int(job["eval_max_steps"])
            and int(provenance.get("step_token_budget") or -1) == 1
            and provenance.get("smiles_class") == job.get("smiles_class")
        )
    except (TypeError, ValueError):
        return False
    if not provenance_matches:
        return False
    dataset = str(job["dataset"])
    if dataset in {"gsm_symbolic", "spider"}:
        split = payload.get("eval_split") or {}
        prefix = "gsm" if dataset == "gsm_symbolic" else "spider"
        if split.get(f"{prefix}_split_name") != "test":
            return False
        if str(split.get(f"{prefix}_split_file")) != str(job["heldout_split_file"]):
            return False
    return True


def heldout_command(job: dict[str, Any], python: Path, csd: Path) -> list[str]:
    command = [
        str(python),
        "-m",
        "synthesis.scripts.reevaluate_compiled_csd",
        str(csd),
        "--dataset",
        str(job["dataset"]),
        "--eval-model",
        str(job["eval_model"]),
        "--eval-backend",
        "vllm",
        "--device",
        "auto",
        "--sample-size",
        str(job["heldout_sample_size"]),
        "--max-steps",
        str(job["eval_max_steps"]),
        "--step-token-budget",
        "1",
        "--max-seconds-per-example",
        str(job["eval_max_seconds"]),
        "--vllm-gpu-memory-utilization",
        str(job["gpu_mem_util"]),
        "--vllm-tensor-parallel-size",
        "1",
        "--output-json",
        str(job["heldout_output_json"]),
        "--provenance-cell-id",
        str(job["cell_id"]),
        "--provenance-manifest-commit",
        str(job["git_commit"]),
    ]
    split_file = str(job.get("heldout_split_file", "")).strip()
    if job["dataset"] == "gsm_symbolic":
        command.extend(["--gsm-split-file", split_file, "--gsm-split-name", "test"])
    elif job["dataset"] == "spider":
        command.extend(["--spider-split-file", split_file, "--spider-split-name", "test"])
    else:
        command.extend(["--smiles-classes", str(job["smiles_class"])])
    return command


def load_manifest(path: Path) -> tuple[str, list[dict[str, Any]]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"invalid manifest {path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("jobs"), list):
        raise ConfigError("manifest must be an object with a jobs list")
    commit = str(payload.get("git_commit", "")).strip()
    if len(commit) != 40:
        raise ConfigError("manifest git_commit must be a full 40-character hash")
    jobs = payload["jobs"]
    if not jobs:
        raise ConfigError("manifest jobs must not be empty")
    seen: set[str] = set()
    required = {
        "cell_id", "task", "dataset", "eval_model", "max_iterations",
        "interrupted_author_calls",
        "eval_sample_size", "min_accuracy", "min_syntax_rate", "eval_max_steps",
        "eval_max_seconds", "memory_reservation_mib", "gpu_mem_util", "output_name",
        "heldout_sample_size", "heldout_split_name", "heldout_output_json",
    }
    for job in jobs:
        missing = sorted(required - job.keys())
        if missing:
            raise ConfigError(f"manifest entry missing {missing}")
        cell = str(job["cell_id"])
        if cell in seen:
            raise ConfigError(f"duplicate cell_id: {cell}")
        seen.add(cell)
        if not str(job["output_name"]).startswith("coldq_"):
            raise ConfigError(f"cold output_name required for {cell}")
        if not 1 <= int(job["max_iterations"]) <= 80:
            raise ConfigError(f"unapproved iteration cap for {cell}")
        if int(job["interrupted_author_calls"]) < 0:
            raise ConfigError(f"negative interrupted author-call count for {cell}")
    return commit, jobs


def validate_baseline_evidence(cell: str, job: dict[str, Any], repo: Path) -> None:
    evidence_path = Path(str(job["baseline_source"]))
    if not evidence_path.is_absolute():
        evidence_path = repo / evidence_path
    try:
        payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"{cell} baseline evidence is invalid: {exc}") from exc
    entry = (payload.get("cells") or {}).get(cell) or {}
    if entry.get("pending_validity") or job.get("baseline_pending_validity"):
        try:
            pending_counts_match = (
                int(entry.get("num_correct")) == int(job["baseline_num_correct"])
                and int(entry.get("num_examples")) == int(job["baseline_num_examples"])
            )
        except (TypeError, ValueError):
            pending_counts_match = False
        if not pending_counts_match:
            raise ConfigError(f"{cell} pending baseline counts do not match manifest")
        return
    try:
        counts_match = (
            int(entry.get("num_correct")) == int(job["baseline_num_correct"])
            and int(entry.get("num_examples")) == int(job["baseline_num_examples"])
        )
    except (TypeError, ValueError):
        counts_match = False
    if not counts_match:
        raise ConfigError(f"{cell} baseline evidence does not match manifest counts")
    expected_strategy = BASELINE_STRATEGY_BY_DATASET[str(job["dataset"])]
    if entry.get("baseline_strategy") != expected_strategy:
        raise ConfigError(
            f"{cell} has wrong baseline strategy; expected {expected_strategy}"
        )
    if (
        entry.get("dataset") != job["dataset"]
        or entry.get("eval_model") != job["eval_model"]
        or entry.get("split_name") != "train"
    ):
        raise ConfigError(f"{cell} baseline evidence has wrong model, dataset, or split")
    artifact = Path(str(entry.get("source_artifact", "")))
    if not artifact.is_absolute():
        artifact = repo / artifact
    if not artifact.is_file():
        raise ConfigError(f"{cell} baseline source artifact is missing: {artifact}")
    if hashlib.sha256(artifact.read_bytes()).hexdigest() != entry.get("source_sha256"):
        raise ConfigError(f"{cell} baseline source artifact hash does not match evidence")
    try:
        source_payload = json.loads(artifact.read_text(encoding="utf-8"))
        source_counts_match = (
            int(source_payload.get("num_correct"))
            == int(job["baseline_num_correct"])
            and int(source_payload.get("num_examples"))
            == int(job["baseline_num_examples"])
        )
    except (OSError, json.JSONDecodeError, AttributeError, TypeError, ValueError):
        source_payload = {}
        source_counts_match = False
    if (
        not source_counts_match
        or source_payload.get("dataset") != job["dataset"]
        or source_payload.get("eval_model") != job["eval_model"]
        or source_payload.get("split_name") != "train"
        or source_payload.get("baseline_strategy") != expected_strategy
    ):
        raise ConfigError(
            f"{cell} baseline source artifact does not support manifest measurement"
        )
    expected_split = CANONICAL_TRAIN_SPLIT_BY_DATASET.get(str(job["dataset"]))
    if expected_split is not None:
        source_split = Path(str(source_payload.get("split_file", "")))
        canonical_split = repo / expected_split
        if (
            source_split != expected_split
            or not canonical_split.is_file()
            or hashlib.sha256(canonical_split.read_bytes()).hexdigest()
            != source_payload.get("split_file_sha256")
        ):
            raise ConfigError(f"{cell} baseline source has wrong canonical train split")
    raw_artifact = Path(str(source_payload.get("raw_source_artifact", "")))
    if not raw_artifact.is_absolute():
        raw_artifact = repo / raw_artifact
    if not raw_artifact.is_file():
        raise ConfigError(f"{cell} raw baseline source artifact is missing: {raw_artifact}")
    if (
        hashlib.sha256(raw_artifact.read_bytes()).hexdigest()
        != source_payload.get("raw_source_sha256")
    ):
        raise ConfigError(f"{cell} raw baseline source artifact hash is invalid")


def validate_heldout_split(cell: str, job: dict[str, Any], repo: Path) -> None:
    expected_relative = CANONICAL_TRAIN_SPLIT_BY_DATASET.get(str(job["dataset"]))
    if expected_relative is None:
        return
    canonical = repo / expected_relative
    requested = Path(str(job.get("heldout_split_file", "")))
    if not requested.is_absolute():
        requested = repo / requested
    if (
        not canonical.is_file()
        or not requested.is_file()
        or requested.resolve() != canonical.resolve()
        or hashlib.sha256(requested.read_bytes()).hexdigest()
        != hashlib.sha256(canonical.read_bytes()).hexdigest()
    ):
        raise ConfigError(
            f"{cell} requires canonical heldout split {expected_relative}"
        )


def validate_exhaustive_campaign(
    jobs: list[dict[str, Any]], *, repo: Path | None = None
) -> None:
    by_cell = {str(job["cell_id"]): job for job in jobs}
    if set(by_cell) != set(EXPECTED_CELLS) or len(jobs) != len(EXPECTED_CELLS):
        raise ConfigError("manifest must contain exactly the 20 approved cells")
    output_names = [str(job["output_name"]) for job in jobs]
    heldout_outputs = [str(job["heldout_output_json"]) for job in jobs]
    if len(set(output_names)) != len(output_names):
        raise ConfigError("output_name values must be unique")
    if len(set(heldout_outputs)) != len(heldout_outputs):
        raise ConfigError("heldout_output_json values must be unique")
    for cell, expected in EXPECTED_CELLS.items():
        job = by_cell[cell]
        for field, value in expected.items():
            if job.get(field) != value:
                raise ConfigError(f"{cell} requires {field}={value!r}")
        expected_runtime = EXPECTED_RUNTIME_BY_MODEL[str(job["eval_model"])]
        for field, value in expected_runtime.items():
            if job.get(field) != value:
                raise ConfigError(f"{cell} requires {field}={value!r}")
        if float(job.get("eval_max_seconds", -1)) != 600.0:
            raise ConfigError(f"{cell} requires eval_max_seconds=600")
        if any(str(field).startswith("initial") for field in job):
            raise ConfigError(f"warm-start field is forbidden for {cell}")
        if float(job.get("min_syntax_rate", -1)) != 0.9:
            raise ConfigError(f"{cell} requires min_syntax_rate=0.9")
        try:
            correct = int(job["baseline_num_correct"])
            total = int(job["baseline_num_examples"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigError(f"{cell} requires integer baseline counts") from exc
        if total != int(job["eval_sample_size"]) or not 0 <= correct < total:
            raise ConfigError(f"{cell} baseline counts must match the train sample size")
        strict_bar = (correct + 1) / total
        if abs(float(job["min_accuracy"]) - strict_bar) > 1e-12:
            raise ConfigError(f"{cell} min_accuracy must be the one-example strict train bar")
        baseline_source = str(job.get("baseline_source", "")).strip()
        if not baseline_source:
            raise ConfigError(f"{cell} requires baseline_source evidence")
        if repo is not None:
            baseline_path = Path(baseline_source)
            if not baseline_path.is_absolute():
                baseline_path = repo / baseline_path
            if not baseline_path.is_file():
                raise ConfigError(f"{cell} baseline_source file is missing: {baseline_path}")
            validate_baseline_evidence(cell, job, repo)
        if str(job.get("heldout_split_name")) != "test":
            raise ConfigError(f"{cell} requires heldout_split_name='test'")
        if not str(job["output_name"]).startswith(f"coldq_{cell}_"):
            raise ConfigError(f"{cell} output_name must be isolated and cold")
        expected_log = Path("outputs") / "generated" / str(job["output_name"]) / "run.log"
        if Path(str(job.get("log_file", ""))) != expected_log:
            raise ConfigError(f"{cell} log_file must be {expected_log}")
        dataset = str(job["dataset"])
        if dataset == "gsm_symbolic":
            expected_split = "gsm_symbolic_crane_proportional_49x49_seed123.json"
        elif dataset == "spider":
            expected_split = "spider_dev_proportional_300x300_seed334.json"
        else:
            expected_split = None
        if expected_split and Path(str(job.get("heldout_split_file", ""))).name != expected_split:
            raise ConfigError(f"{cell} requires canonical heldout split {expected_split}")
        if expected_split and repo is not None:
            validate_heldout_split(cell, job, repo)
    total_author_calls = sum(
        int(job["max_iterations"]) + int(job["interrupted_author_calls"])
        for job in jobs
    )
    if total_author_calls != APPROVED_AUTHOR_CALL_CAP:
        raise ConfigError(
            "campaign author-call accounting must total "
            f"{APPROVED_AUTHOR_CALL_CAP}, got {total_author_calls}"
        )


def _write_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_job(
    job: dict[str, Any],
    gpus: tuple[int, ...],
    *,
    repo: Path,
    python: Path,
    state_dir: Path,
) -> int:
    cell = str(job["cell_id"])
    primary_gpu = gpus[0]
    gpu_list = ",".join(str(gpu) for gpu in gpus)
    output_name = str(job["output_name"])
    output_root = repo / "outputs" / "generated" / output_name
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / "run.log"
    state_path = state_dir / f"{cell}.json"
    heldout_path = Path(str(job["heldout_output_json"]))
    if heldout_is_complete(heldout_path, job):
        logger.warning("[coldq] already complete cell=%s heldout=%s", cell, heldout_path)
        return 0
    csd = compiled_csd(
        repo,
        output_name,
        min_accuracy=float(job["min_accuracy"]),
        min_syntax_rate=float(job["min_syntax_rate"]),
        job=job,
    )
    synthesis_exhausted = csd is not None and synthesis_was_exhausted(
        repo, output_name, job
    )
    combined_log_path = Path(
        os.environ.get(
            "CSD_COMBINED_RUN_LOG",
            str(repo / "logs" / "coldq_combined_run.log"),
        )
    )
    combined_log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log, combined_log_path.open(
        "a", encoding="utf-8"
    ) as combined_log:
        combined_log.write(f"\n===== CELL {cell} BEGIN =====\n")
        combined_log.flush()
        _human_log_emit(repo, cell, "COLDQ_CELL_BEGIN", f"gpus={gpu_list}")
        tee = _CombinedLogWriter(log, combined_log)
        if csd is None:
            previous_run_dir = current_run_dir(repo, output_name)
            logger.warning(
                "[coldq] synthesis start cell=%s gpus=%s workers=%d output=%s",
                cell,
                gpu_list,
                len(gpus),
                output_name,
            )
            tee.write(
                f"COLDQ_SYNTHESIS_START cell={cell} gpus={gpu_list} "
                f"workers={len(gpus)} commit={job['git_commit']}\n"
            )
            tee.flush()
            _human_log_emit(
                repo,
                cell,
                "COLDQ_SYNTHESIS_START",
                f"gpus={gpu_list} workers={len(gpus)}",
            )
            status = _run_command_teeing_stdout(
                synthesis_command(job, python),
                cwd=repo,
                env=synthesis_environment(job, gpus, os.environ, repo),
                streams=(log, combined_log),
            )
            tee.write(f"COLDQ_SYNTHESIS_FINISH cell={cell} status={status}\n")
            tee.flush()
            _human_log_emit(repo, cell, "COLDQ_SYNTHESIS_FINISH", f"status={status}")
            synthesis_exhausted = status == 1 and synthesis_was_exhausted(
                repo,
                output_name,
                job,
                previous_run_dir=previous_run_dir,
            )
            if status not in {0, 1}:
                _write_state(state_path, {"cell_id": cell, "status": "error", "exit_code": status})
                return status
            if status == 1 and not synthesis_exhausted:
                _write_state(state_path, {"cell_id": cell, "status": "error", "exit_code": status})
                return status
            csd = compiled_csd(
                repo,
                output_name,
                min_accuracy=float(job["min_accuracy"]),
                min_syntax_rate=float(job["min_syntax_rate"]),
                job=job,
            )
        if csd is None:
            status = "complete_failure" if synthesis_exhausted else "error"
            _write_state(state_path, {"cell_id": cell, "status": status, "reason": "missing_csd"})
            return TERMINAL_SYNTHESIS_FAILURE if synthesis_exhausted else 3
        if synthesis_exhausted:
            logger.warning("[coldq] using best exhausted attempt cell=%s csd=%s", cell, csd)
        heldout_path.parent.mkdir(parents=True, exist_ok=True)
        logger.warning(
            "[coldq] heldout start cell=%s gpu=%d csd=%s",
            cell,
            primary_gpu,
            csd,
        )
        status = _run_command_teeing_stdout(
            heldout_command(job, python, csd),
            cwd=repo,
            env=author_free_environment(
                os.environ, primary_gpu, dataset=str(job["dataset"])
            ),
            streams=(log, combined_log),
        )
    heldout_complete = status == 0 and heldout_is_complete(heldout_path, job)
    if status == 0 and not heldout_complete:
        status = 4
    state = "complete_loss" if synthesis_exhausted and heldout_complete else (
        "complete_success" if heldout_complete else "error"
    )
    _write_state(state_path, {"cell_id": cell, "status": state, "heldout": str(heldout_path)})
    return status


def dispatch(jobs, *, snapshot, worker, poll_seconds: float) -> None:
    pending = [dict(job) for job in jobs]
    reservations: dict[int, dict[str, int]] = {}
    running: dict[concurrent.futures.Future[int], tuple[tuple[int, ...], str]] = {}
    failures: list[tuple[str, int]] = []
    baseline = snapshot()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        while pending or running:
            snapshots = snapshot()
            for gpu in snapshots:
                reservations.setdefault(gpu, {})
                if not reservations[gpu]:
                    baseline[gpu] = dict(snapshots[gpu])
            launched = True
            while pending and launched:
                launched = False
                for index, job in enumerate(pending):
                    gpus = choose_gpu_bundle(job, snapshots, reservations, baseline)
                    if gpus is None:
                        continue
                    cell = str(job["cell_id"])
                    for gpu in gpus:
                        reservations[gpu][cell] = synthesis_required_memory_mib(
                            job, snapshots[gpu]["total_mib"]
                        )
                    logger.warning(
                        "[coldq] dispatch cell=%s gpus=%s workers=%d",
                        cell,
                        ",".join(str(gpu) for gpu in gpus),
                        len(gpus),
                    )
                    running[executor.submit(worker, job, gpus)] = (gpus, cell)
                    pending.pop(index)
                    launched = True
                    break
            finished = [future for future in running if future.done()]
            for future in finished:
                gpus, cell = running.pop(future)
                status = future.result()
                for gpu in gpus:
                    reservations[gpu].pop(cell, None)
                logger.warning(
                    "[coldq] release cell=%s gpus=%s status=%d",
                    cell,
                    ",".join(str(gpu) for gpu in gpus),
                    status,
                )
                if status not in {0, TERMINAL_SYNTHESIS_FAILURE}:
                    failures.append((cell, status))
            if pending and not running and not finished:
                # Temporary contention (other users / Phase1 / twins) should wait,
                # not abort the whole campaign. free_mib gate makes this common.
                logger.warning(
                    "[coldq] no queued cold job fits available GPU memory; waiting %.0fs",
                    poll_seconds,
                )
            if pending or running:
                time.sleep(max(0.0, poll_seconds))
    if failures:
        raise ConfigError("worker failures: " + ", ".join(f"{cell}:{status}" for cell, status in failures))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--lock-file", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--repair-attestation", type=Path)
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument(
        "--exclude-cell-prefix",
        action="append",
        default=[],
        help=(
            "After validating the full 20-cell manifest, skip dispatch for cell_ids "
            "with any of these prefixes (repeatable). Used to keep GSM blocked."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
    try:
        commit, jobs = load_manifest(args.manifest)
        validate_exhaustive_campaign(jobs, repo=args.repo)
        launch_commit = verify_repo_version(args.repo, commit, args.repair_attestation)
        for job in jobs:
            job["git_commit"] = commit
            job["launch_commit"] = launch_commit
        prefixes = [str(p) for p in args.exclude_cell_prefix if str(p)]
        if prefixes:
            excluded = [
                str(job["cell_id"])
                for job in jobs
                if any(str(job["cell_id"]).startswith(prefix) for prefix in prefixes)
            ]
            jobs = [
                job
                for job in jobs
                if not any(str(job["cell_id"]).startswith(prefix) for prefix in prefixes)
            ]
            logger.warning(
                "[coldq] exclude prefixes=%s excluded=%s remaining=%d",
                ",".join(prefixes),
                ",".join(excluded),
                len(jobs),
            )
            if not jobs:
                raise ConfigError("all cells excluded; nothing to dispatch")
        args.lock_file.parent.mkdir(parents=True, exist_ok=True)
        with args.lock_file.open("w", encoding="utf-8") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ConfigError("another cold queue controller is active") from exc
            def worker(job: dict[str, Any], gpus: tuple[int, ...]) -> int:
                return run_job(
                    job,
                    gpus,
                    repo=args.repo,
                    python=args.python,
                    state_dir=args.state_dir,
                )
            if args.dry_run:
                for job in jobs:
                    logger.warning(
                        "[coldq] dry-run cell=%s required_gpus=%d command=%r",
                        job["cell_id"],
                        required_gpu_count(job),
                        synthesis_command(job, args.python),
                    )
            else:
                dispatch(
                    jobs,
                    snapshot=lambda: gpu_memory_snapshot(args.nvidia_smi),
                    worker=worker,
                    poll_seconds=args.poll_seconds,
                )
        return 0
    except (ConfigError, subprocess.CalledProcessError) as exc:
        logger.error("[coldq] configuration error: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
