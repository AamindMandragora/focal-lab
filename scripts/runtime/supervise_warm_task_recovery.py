#!/usr/bin/env python3
"""Persist and recover the approved focal warm-recovery queue."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.runtime.run_warm_task_recovery_queue import load_manifest


SYNTHESIS_REQUIRED = "synthesis_required"
HELDOUT_REQUIRED = "heldout_required"
COMPLETE_SUCCESS = "complete_success"
COMPLETE_FAILURE = "complete_failure"
LOGGER = logging.getLogger("warm-recovery-supervisor")


def _latest_run(repo: Path, job: dict[str, Any]) -> Path | None:
    pointer = repo / "outputs" / "generated" / str(job["output_name"]) / "latest_run.txt"
    if not pointer.is_file():
        return None
    run_dir = Path(pointer.read_text(encoding="utf-8").strip())
    return run_dir if run_dir.is_absolute() else repo / run_dir


def _valid_json(path: Path) -> bool:
    try:
        return path.is_file() and isinstance(
            json.loads(path.read_text(encoding="utf-8")), dict
        )
    except (OSError, json.JSONDecodeError):
        return False


def _last_reported_attempt(path: Path) -> int | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        attempts = payload.get("attempts", [])
        numbers = [
            int(item["attempt_number"])
            for item in attempts
            if isinstance(item, dict) and "attempt_number" in item
        ]
        return max(numbers) if numbers else None
    except (OSError, ValueError, TypeError, json.JSONDecodeError, AttributeError):
        return None


def _terminal_attempt(payload: dict[str, Any]) -> int | None:
    try:
        return int(payload["evaluated_attempt"])
    except (KeyError, TypeError, ValueError):
        match = re.search(r"attempt cap already evaluated: (\d+)/(\d+)", str(payload.get("detail", "")))
        return int(match.group(1)) if match else None


def job_phase(repo: Path, job: dict[str, Any]) -> str:
    """Classify one row from persisted reports, which are the source of truth."""
    terminal_path = str(job.get("terminal_state_file", "")).strip()
    if terminal_path:
        terminal = Path(terminal_path)
        try:
            payload = json.loads(terminal.read_text(encoding="utf-8"))
            if payload.get("phase") == COMPLETE_FAILURE:
                terminal_attempt = _terminal_attempt(payload)
                total_cap = int(job.get("total_cap", 0))
                if terminal_attempt is None or total_cap <= 0 or terminal_attempt >= total_cap:
                    return COMPLETE_FAILURE
        except (OSError, json.JSONDecodeError, AttributeError):
            pass
    run_dir = _latest_run(repo, job)
    if run_dir is None:
        return SYNTHESIS_REQUIRED
    results = run_dir / "results"
    failure_report = results / "failure_report.json"
    if _valid_json(failure_report):
        total_cap = int(job.get("total_cap", 0))
        last_attempt = _last_reported_attempt(failure_report)
        if total_cap <= 0 or last_attempt is None or last_attempt >= total_cap:
            return COMPLETE_FAILURE
        return SYNTHESIS_REQUIRED
    if not _valid_json(results / "success_report.json"):
        return SYNTHESIS_REQUIRED
    heldout = Path(str(job["heldout_output_json"]))
    return COMPLETE_SUCCESS if _valid_json(heldout) else HELDOUT_REQUIRED


def apply_conditional_extensions(
    repo: Path, jobs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Activate an approved larger cap only after every row in its group fails."""
    effective = [dict(job) for job in jobs]
    groups: dict[str, list[int]] = {}
    for index, job in enumerate(effective):
        group = str(job.get("extension_group", "")).strip()
        if group:
            groups.setdefault(group, []).append(index)
    for group, indexes in groups.items():
        if all(job_phase(repo, effective[index]) == COMPLETE_FAILURE for index in indexes):
            for index in indexes:
                extended_cap = int(effective[index].get("extension_total_cap", 0))
                if extended_cap <= int(effective[index]["total_cap"]):
                    raise ValueError(f"invalid extension cap for {effective[index]['cell_id']}: {extended_cap}")
                effective[index]["total_cap"] = extended_cap
                effective[index]["extension_active"] = group
    return effective


def write_state(
    path: Path,
    jobs: list[dict[str, Any]],
    phases: dict[str, str],
    *,
    controller_pid: int | None,
) -> None:
    """Atomically publish a compact state snapshot for monitoring and restart."""
    payload = {
        "version": 1,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "controller_pid": controller_pid,
        "jobs": {
            str(job["cell_id"]): {
                "phase": phases[str(job["cell_id"])],
                "output_name": str(job["output_name"]),
            }
            for job in jobs
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def controller_pid_from_file(
    path: Path, proc_root: Path, expected_manifest: Path
) -> int | None:
    """Return only a live recovery-controller PID, never a reused stale PID."""
    try:
        pid = int(path.read_text(encoding="utf-8").strip())
        process_dir = proc_root / str(pid)
        arguments = [
            part.decode("utf-8", "replace")
            for part in process_dir.joinpath("cmdline").read_bytes().split(b"\0")
            if part
        ]
        is_controller = any(
            Path(argument).name == "run_warm_task_recovery_queue.py"
            for argument in arguments
        )
        manifest_index = arguments.index("--manifest")
        manifest = Path(arguments[manifest_index + 1])
        if not manifest.is_absolute():
            manifest = process_dir.joinpath("cwd").resolve() / manifest
        return pid if is_controller and manifest.resolve() == expected_manifest.resolve() else None
    except (OSError, ValueError, IndexError, ProcessLookupError):
        return None


def recovery_processes(
    proc_root: Path,
    jobs: list[dict[str, Any]],
    *,
    expected_repo: Path | None = None,
) -> set[int]:
    outputs = {str(job["output_name"]) for job in jobs}
    matches: set[int] = set()
    parents: dict[int, int] = {}
    for process_dir in proc_root.iterdir():
        if not process_dir.name.isdigit():
            continue
        try:
            arguments = [
                part.decode("utf-8", "replace")
                for part in process_dir.joinpath("cmdline").read_bytes().split(b"\0")
                if part
            ]
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        output_names = {
            arguments[index + 1]
            for index, value in enumerate(arguments[:-1])
            if value == "--output-name"
        }
        if outputs & output_names:
            try:
                same_repo = (
                    expected_repo is None
                    or process_dir.joinpath("cwd").resolve() == expected_repo.resolve()
                )
            except (OSError, ProcessLookupError):
                same_repo = False
            if same_repo:
                matches.add(int(process_dir.name))
        try:
            parent_line = next(
                line
                for line in process_dir.joinpath("status").read_text().splitlines()
                if line.startswith("PPid:")
            )
            parents[int(process_dir.name)] = int(parent_line.split(":", 1)[1].strip())
        except (FileNotFoundError, ProcessLookupError, PermissionError, StopIteration, ValueError):
            continue

    added = True
    while added:
        descendants = {pid for pid, parent in parents.items() if parent in matches}
        added = not descendants.issubset(matches)
        matches.update(descendants)
    return matches


def terminate_stale_workers(
    proc_root: Path, jobs: list[dict[str, Any]], *, expected_repo: Path | None = None
) -> None:
    """Stop exact manifest workers left behind by a dead controller."""
    stale = recovery_processes(proc_root, jobs, expected_repo=expected_repo)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if not stale:
            return
        LOGGER.warning("[warm-recovery-supervisor] stopping stale pids=%s", sorted(stale))
        for pid in stale:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        if sig == signal.SIGTERM:
            time.sleep(2)
            stale = {pid for pid in stale if (proc_root / str(pid)).exists()}


def _phases(repo: Path, jobs: list[dict[str, Any]]) -> dict[str, str]:
    return {str(job["cell_id"]): job_phase(repo, job) for job in jobs}


def _write_pending_manifest(
    path: Path, jobs: list[dict[str, Any]], phases: dict[str, str]
) -> list[dict[str, Any]]:
    pending = [
        job
        for job in jobs
        if phases[str(job["cell_id"])] in {SYNTHESIS_REQUIRED, HELDOUT_REQUIRED}
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(pending, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return pending


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--resume-script", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--controller-lock", type=Path, required=True)
    parser.add_argument("--controller-pid-file", type=Path, required=True)
    parser.add_argument("--state-file", type=Path, required=True)
    parser.add_argument("--pending-manifest", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--retry-seconds", type=float, default=3600)
    parser.add_argument("--proc-root", type=Path, default=Path("/proc"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
    jobs = load_manifest(args.manifest)

    adopted_pid = controller_pid_from_file(
        args.controller_pid_file, args.proc_root, args.manifest
    )
    while adopted_pid is not None:
        effective_jobs = apply_conditional_extensions(args.repo, jobs)
        phases = _phases(args.repo, effective_jobs)
        write_state(args.state_file, effective_jobs, phases, controller_pid=adopted_pid)
        LOGGER.warning(
            "[warm-recovery-supervisor] adopting controller_pid=%d unfinished=%d",
            adopted_pid,
            sum(phase in {SYNTHESIS_REQUIRED, HELDOUT_REQUIRED} for phase in phases.values()),
        )
        time.sleep(args.poll_seconds)
        adopted_pid = controller_pid_from_file(
            args.controller_pid_file, args.proc_root, args.manifest
        )

    while True:
        effective_jobs = apply_conditional_extensions(args.repo, jobs)
        phases = _phases(args.repo, effective_jobs)
        write_state(args.state_file, effective_jobs, phases, controller_pid=None)
        pending = _write_pending_manifest(args.pending_manifest, effective_jobs, phases)
        if not pending:
            LOGGER.warning("[warm-recovery-supervisor] all rows terminal")
            return 0

        terminate_stale_workers(args.proc_root, effective_jobs, expected_repo=args.repo)
        command = [
            str(args.python),
            str(args.controller),
            "--repo", str(args.repo),
            "--manifest", str(args.pending_manifest),
            "--resume-script", str(args.resume_script),
            "--python", str(args.python),
            "--lock-file", str(args.controller_lock),
            "--poll-seconds", str(args.poll_seconds),
        ]
        controller = subprocess.Popen(command, cwd=args.repo)
        args.controller_pid_file.write_text(f"{controller.pid}\n", encoding="utf-8")
        LOGGER.warning(
            "[warm-recovery-supervisor] launched controller_pid=%d pending=%d",
            controller.pid,
            len(pending),
        )
        status = controller.wait()
        effective_jobs = apply_conditional_extensions(args.repo, jobs)
        phases = _phases(args.repo, effective_jobs)
        write_state(args.state_file, effective_jobs, phases, controller_pid=None)
        unfinished = sum(
            phase in {SYNTHESIS_REQUIRED, HELDOUT_REQUIRED}
            for phase in phases.values()
        )
        if not unfinished:
            LOGGER.warning("[warm-recovery-supervisor] queue complete")
            return 0
        LOGGER.warning(
            "[warm-recovery-supervisor] controller exited status=%d unfinished=%d; retrying in %.0fs",
            status,
            unfinished,
            args.retry_seconds,
        )
        time.sleep(args.retry_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
