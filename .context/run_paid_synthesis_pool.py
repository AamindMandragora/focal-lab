#!/usr/bin/env python3
"""Greedily schedule an approval-bound synthesis job set across focal GPUs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import NamedTuple


class Job(NamedTuple):
    label: str
    estimated_memory_mib: int


class Assignment(NamedTuple):
    job: Job
    gpu_id: int


FIXED_JOBS = (
    Job("smiles-qwen35-9b-acrylates", 24_576),
    Job("smiles-qwen35-9b-chain_extenders", 24_576),
    Job("smiles-qwen35-4b-acrylates", 18_432),
    Job("smiles-qwen35-4b-chain_extenders", 18_432),
    Job("gsm14b", 32_768),
    Job("spider14b", 32_768),
    Job("smiles-qwen35-2b-chain_extenders", 16_384),
)
INFRA_RETRY_JOBS = (
    Job("smiles-qwen35-4b-acrylates", 18_432),
    Job("smiles-qwen35-4b-chain_extenders", 18_432),
    Job("gsm14b", 32_768),
    Job("spider14b", 32_768),
)
JOBS = FIXED_JOBS

APPROVAL_CONTRACTS = {
    "fixed": {
        "marker": "User approval is explicit for the seven fixed paid Bedrock synthesis jobs",
        "cost_min": 140,
        "cost_max": 420,
        "approval": "saved-results/2026-07-11-fixed-paid-synthesis-approval.json",
    },
    "infra-retry": {
        "marker": "User approval is explicit for the four fixed infrastructure-retry Bedrock synthesis jobs",
        "cost_min": 80,
        "cost_max": 240,
        "approval": "saved-results/2026-07-11-fixed-infra-retry-approval.json",
    },
}


def jobs_for_set(job_set: str) -> tuple[Job, ...]:
    if job_set == "fixed":
        return FIXED_JOBS
    if job_set == "infra-retry":
        return INFRA_RETRY_JOBS
    raise ValueError(f"unknown paid job set: {job_set}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def env_value(path: Path, key: str) -> str:
    if not path.is_file():
        return ""
    prefix = f"{key}="
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :].strip().strip('"').strip("'")
    return ""


def verify_approval(
    approval_path: Path,
    *,
    queue_script: Path,
    pool_script: Path,
    run_synth: Path,
    env_path: Path,
    job_set: str = "fixed",
) -> None:
    try:
        approval = json.loads(approval_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid fixed paid approval: {error}") from error
    contract = APPROVAL_CONTRACTS[job_set]
    expected_labels = sorted(job.label for job in jobs_for_set(job_set))
    checks = {
        "approval_marker": contract["marker"],
        "account_id": "887730490125",
        "region": "us-east-1",
        "cell_count": len(expected_labels),
        "approved_cells": expected_labels,
        "estimated_cost_min_usd": contract["cost_min"],
        "estimated_cost_max_usd": contract["cost_max"],
        "max_approved_cost_usd": contract["cost_max"],
        "queue_sha256": sha256(queue_script),
        "pool_sha256": sha256(pool_script),
        "run_synth_sha256": sha256(run_synth),
    }
    for field, expected in checks.items():
        if approval.get(field) != expected:
            raise RuntimeError(
                f"fixed paid approval mismatch for {field}: expected {expected!r}"
            )
    if env_value(env_path, "AWS_REGION") != "us-east-1":
        raise RuntimeError("focal .env AWS_REGION does not match approved us-east-1")
    if not env_value(env_path, "AWS_BEARER_TOKEN_BEDROCK"):
        raise RuntimeError("focal .env has no nonempty AWS_BEARER_TOKEN_BEDROCK")


def parse_gpu_ids(value: str) -> list[int]:
    gpu_ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not gpu_ids or len(gpu_ids) != len(set(gpu_ids)):
        raise argparse.ArgumentTypeError("GPU ids must be a unique nonempty list")
    return gpu_ids


def plan_initial_wave(
    jobs: tuple[Job, ...] | list[Job],
    *,
    gpu_ids: list[int],
    total_memory_mib: int,
    existing_memory_mib: dict[int, int],
    safety_memory_mib: int,
) -> tuple[list[Assignment], list[Job]]:
    capacity = total_memory_mib - safety_memory_mib
    projected = {gpu_id: existing_memory_mib[gpu_id] for gpu_id in gpu_ids}
    assignments: list[Assignment] = []
    pending: list[Job] = []
    for job in jobs:
        fitting = [
            gpu_id
            for gpu_id in gpu_ids
            if projected[gpu_id] + job.estimated_memory_mib <= capacity
        ]
        if not fitting:
            pending.append(job)
            continue
        gpu_id = min(fitting, key=lambda item: (projected[item], item))
        assignments.append(Assignment(job, gpu_id))
        projected[gpu_id] += job.estimated_memory_mib
    return assignments, pending


def worker_start_max_used_mib(
    job: Job, *, total_memory_mib: int, safety_memory_mib: int
) -> int:
    """Maximum existing use that still leaves this worker's reservation free."""
    return total_memory_mib - safety_memory_mib - job.estimated_memory_mib


def projected_used_mib(
    *, current_memory_mib: int, baseline_memory_mib: int, reserved_memory_mib: int
) -> int:
    """Count external baseline plus reservations without double-counting loaded jobs."""
    return max(current_memory_mib, baseline_memory_mib + reserved_memory_mib)


def ensure_infra_retry_idle(tmux_bin: str, pgrep_bin: str) -> None:
    """Refuse to overlap the repair cycle with the still-running fixed synthesis pool."""
    tmux_result = subprocess.run(
        [tmux_bin, "has-session", "-t", "paid_synthesis_queue"],
        text=True,
        capture_output=True,
    )
    if tmux_result.returncode == 0:
        raise RuntimeError("paid_synthesis_queue is still active; infra retry must wait")
    if tmux_result.returncode not in {0, 1}:
        raise RuntimeError(f"tmux precondition check failed: {tmux_result.stderr.strip()}")
    pgrep_result = subprocess.run(
        [pgrep_bin, "-u", str(os.getuid()), "-f", "synthesis.run_synthesis"],
        text=True,
        capture_output=True,
    )
    if pgrep_result.returncode == 0:
        raise RuntimeError("an aadivyar synthesis.run_synthesis process is still active")
    if pgrep_result.returncode != 1:
        raise RuntimeError(f"pgrep precondition check failed: {pgrep_result.stderr.strip()}")


def ensure_infra_retry_worker_safe(
    tmux_bin: str,
    pgrep_bin: str,
    *,
    status: str,
    claims_dir: str,
) -> None:
    if status != "logs/paid_synthesis_infra_retry_queue_status.tsv":
        raise RuntimeError("infra retry worker must use the separate retry status path")
    if claims_dir != ".context/paid_synthesis_infra_retry_claims":
        raise RuntimeError("infra retry worker must use the separate retry claims path")
    tmux_result = subprocess.run(
        [tmux_bin, "has-session", "-t", "paid_synthesis_queue"],
        text=True,
        capture_output=True,
    )
    if tmux_result.returncode == 0:
        raise RuntimeError("paid_synthesis_queue is still active; infra retry worker refused")
    if tmux_result.returncode != 1:
        raise RuntimeError(f"tmux worker check failed: {tmux_result.stderr.strip()}")
    pgrep_result = subprocess.run(
        [pgrep_bin, "-u", str(os.getuid()), "-af", "synthesis.run_synthesis"],
        text=True,
        capture_output=True,
    )
    if pgrep_result.returncode == 0:
        active_lines = [line for line in pgrep_result.stdout.splitlines() if line.strip()]
        if not active_lines or any("_infraretry_0711" not in line for line in active_lines):
            raise RuntimeError("a non-retry synthesis process is still active")
    elif pgrep_result.returncode != 1:
        raise RuntimeError(f"pgrep worker check failed: {pgrep_result.stderr.strip()}")


def query_gpu_memory(nvidia_smi: str, gpu_ids: list[int]) -> dict[int, int]:
    result = subprocess.run(
        [
            nvidia_smi,
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    wanted = set(gpu_ids)
    memory: dict[int, int] = {}
    for line in result.stdout.splitlines():
        index_text, used_text = [part.strip() for part in line.split(",", 1)]
        index = int(index_text)
        if index in wanted:
            memory[index] = int(used_text)
    if set(memory) != wanted:
        raise RuntimeError(f"nvidia-smi omitted GPUs: {sorted(wanted - set(memory))}")
    return memory


def write_state(path: Path, *, pending: deque[Job], running: dict[int, tuple[Assignment, subprocess.Popen]], terminal: dict[str, int]) -> None:
    payload = {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pending": [job.label for job in pending],
        "running": [
            {"label": assignment.job.label, "gpu": assignment.gpu_id, "pid": pid}
            for pid, (assignment, _process) in sorted(running.items())
        ],
        "terminal": terminal,
    }
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("/home/aadivyar/csd-generation"))
    parser.add_argument("--gpu-ids", type=parse_gpu_ids, default=parse_gpu_ids("0,1,2,3"))
    parser.add_argument("--total-memory-mib", type=int, default=40_960)
    parser.add_argument("--safety-memory-mib", type=int, default=2_000)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    parser.add_argument("--tmux", default="tmux")
    parser.add_argument("--pgrep", default="pgrep")
    parser.add_argument("--job-set", choices=sorted(APPROVAL_CONTRACTS), default="fixed")
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--verify-worker-launch", action="store_true")
    parser.add_argument("--worker-status", default="")
    parser.add_argument("--worker-claims-dir", default="")
    args = parser.parse_args()

    repo = args.repo.resolve()
    jobs = jobs_for_set(args.job_set)
    queue_script = repo / ".context" / "run_paid_synthesis_queue.sh"
    run_synth = repo / "run_synth_cell.sh"
    if args.job_set == "fixed":
        state_path = repo / "logs" / "paid_synthesis_pool_state.json"
        claim_path = repo / ".context" / "paid_synthesis_pool.claim"
        status_path = repo / "logs" / "paid_synthesis_queue_status.tsv"
        claims_dir = repo / ".context" / "paid_synthesis_fixed_claims"
        completion_path = repo / "logs" / "paid_synthesis_fixed_complete.json"
    else:
        state_path = repo / "logs" / "paid_synthesis_infra_retry_pool_state.json"
        claim_path = repo / ".context" / "paid_synthesis_infra_retry_pool.claim"
        status_path = repo / "logs" / "paid_synthesis_infra_retry_queue_status.tsv"
        claims_dir = repo / ".context" / "paid_synthesis_infra_retry_claims"
        completion_path = repo / "logs" / "paid_synthesis_infra_retry_complete.json"
    repo.joinpath("logs").mkdir(parents=True, exist_ok=True)

    if not args.dry_run:
        requested_approval = args.approval or Path(APPROVAL_CONTRACTS[args.job_set]["approval"])
        approval_path = requested_approval if requested_approval.is_absolute() else repo / requested_approval
        verify_approval(
            approval_path,
            queue_script=queue_script,
            pool_script=Path(__file__).resolve(),
            run_synth=run_synth,
            env_path=repo / ".env",
            job_set=args.job_set,
        )
    if args.verify_worker_launch:
        if args.job_set != "infra-retry":
            raise SystemExit("--verify-worker-launch is only valid for infra-retry")
        ensure_infra_retry_worker_safe(
            args.tmux,
            args.pgrep,
            status=args.worker_status,
            claims_dir=args.worker_claims_dir,
        )
        print("[paid-synthesis-pool] infra retry worker verified", flush=True)
        return 0
    if args.verify_only:
        print("[paid-synthesis-pool] approval verified", flush=True)
        return 0
    if args.job_set == "infra-retry" and not args.dry_run:
        ensure_infra_retry_idle(args.tmux, args.pgrep)

    memory = (
        {gpu_id: 0 for gpu_id in args.gpu_ids}
        if args.dry_run
        else query_gpu_memory(args.nvidia_smi, args.gpu_ids)
    )
    initial, initially_pending = plan_initial_wave(
        jobs,
        gpu_ids=args.gpu_ids,
        total_memory_mib=args.total_memory_mib,
        existing_memory_mib=memory,
        safety_memory_mib=args.safety_memory_mib,
    )
    if args.dry_run:
        for assignment in initial:
            print(
                f"DRY_RUN_ASSIGN label={assignment.job.label} gpu={assignment.gpu_id} "
                f"estimated_memory_mib={assignment.job.estimated_memory_mib}"
            )
        for job in initially_pending:
            print(f"DRY_RUN_PENDING label={job.label}")
        return 0

    baseline_memory = dict(memory)

    if not queue_script.is_file():
        raise SystemExit(f"missing fixed queue script: {queue_script}")
    try:
        claim_path.mkdir()
    except FileExistsError:
        raise SystemExit(f"paid synthesis pool is already claimed: {claim_path}")

    pending = deque(jobs)
    running: dict[int, tuple[Assignment, subprocess.Popen]] = {}
    reserved = {gpu_id: 0 for gpu_id in args.gpu_ids}
    terminal: dict[str, int] = {}
    interrupted = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    try:
        while pending or running:
            for pid, (assignment, process) in list(running.items()):
                return_code = process.poll()
                if return_code is None:
                    continue
                terminal[assignment.job.label] = return_code
                reserved[assignment.gpu_id] -= assignment.job.estimated_memory_mib
                del running[pid]
                print(
                    f"[paid-synthesis-pool] finish label={assignment.job.label} "
                    f"gpu={assignment.gpu_id} exit={return_code}",
                    flush=True,
                )

            if interrupted:
                print(
                    "[paid-synthesis-pool] interrupted; leaving active children and claim in place",
                    file=sys.stderr,
                    flush=True,
                )
                write_state(state_path, pending=pending, running=running, terminal=terminal)
                return 130

            memory = query_gpu_memory(args.nvidia_smi, args.gpu_ids)
            candidates = len(pending)
            for _ in range(candidates):
                job = pending.popleft()
                capacity = args.total_memory_mib - args.safety_memory_mib
                fitting = [
                    gpu_id
                    for gpu_id in args.gpu_ids
                    if projected_used_mib(
                        current_memory_mib=memory[gpu_id],
                        baseline_memory_mib=baseline_memory[gpu_id],
                        reserved_memory_mib=reserved[gpu_id],
                    )
                    + job.estimated_memory_mib
                    <= capacity
                ]
                if not fitting:
                    pending.append(job)
                    continue
                gpu_id = min(
                    fitting,
                    key=lambda item: (
                        projected_used_mib(
                            current_memory_mib=memory[item],
                            baseline_memory_mib=baseline_memory[item],
                            reserved_memory_mib=reserved[item],
                        ),
                        item,
                    ),
                )
                environment = {
                    **os.environ,
                    "REPO": str(repo),
                    "GPU": str(gpu_id),
                    "ONLY_LABEL": job.label,
                    "PAID_JOB_SET": args.job_set,
                    "STATUS": str(status_path.relative_to(repo)),
                    "CLAIMS_DIR": str(claims_dir.relative_to(repo)),
                    "FIXED_COMPLETE": str(completion_path.relative_to(repo)),
                    "GPU_WAIT_MAX_USED_MIB": str(
                        worker_start_max_used_mib(
                            job,
                            total_memory_mib=args.total_memory_mib,
                            safety_memory_mib=args.safety_memory_mib,
                        )
                    ),
                    "PYTHONUNBUFFERED": "1",
                }
                process = subprocess.Popen(
                    ["bash", str(queue_script)],
                    cwd=repo,
                    env=environment,
                    start_new_session=True,
                )
                assignment = Assignment(job, gpu_id)
                running[process.pid] = (assignment, process)
                reserved[gpu_id] += job.estimated_memory_mib
                print(
                    f"[paid-synthesis-pool] launch label={job.label} gpu={gpu_id} "
                    f"pid={process.pid} estimate_mib={job.estimated_memory_mib}",
                    flush=True,
                )

            write_state(state_path, pending=pending, running=running, terminal=terminal)
            if pending or running:
                time.sleep(args.poll_seconds)

        finalize = subprocess.run(
            ["bash", str(queue_script)],
            cwd=repo,
            env={
                **os.environ,
                "REPO": str(repo),
                "FINALIZE_ONLY": "1",
                "PAID_JOB_SET": args.job_set,
                "STATUS": str(status_path.relative_to(repo)),
                "CLAIMS_DIR": str(claims_dir.relative_to(repo)),
                "FIXED_COMPLETE": str(completion_path.relative_to(repo)),
            },
        )
        if finalize.returncode != 0:
            return finalize.returncode
        claim_path.rmdir()
        print("[paid-synthesis-pool] complete", flush=True)
        return 0
    except BaseException:
        write_state(state_path, pending=pending, running=running, terminal=terminal)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
