#!/usr/bin/env python3
"""Detect catastrophic recovery failures, ask Claude Code for a bounded fix, and verify it."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.runtime.run_cold_synthesis_queue import load_manifest as load_cold_manifest
from scripts.runtime.run_warm_task_recovery_queue import load_manifest as load_warm_manifest
from scripts.runtime.supervise_warm_task_recovery import recovery_processes

LOGGER = logging.getLogger("codex-incident-monitor")

RULES = (
    ("unknown_task", ("unknown task", "task_chars=0")),
    ("gpu_out_of_memory", ("cuda out of memory",)),
    (
        "gpu_memory_startup",
        ("engine core initialization failed", "valueerror: free memory on device"),
    ),
    ("disk_full", ("no space left on device",)),
    (
        "configuration_error",
        ("[warm-recovery] configuration error", "[coldq] configuration error"),
    ),
    (
        "missing_persisted_csd",
        ("synthesis returned success without persisted csd",),
    ),
    (
        "worker_failed",
        (
            "[warm-recovery] done status=",
            "[warm-recovery] release cell=",
            "[coldq] release cell=",
        ),
    ),
    (
        "claude_timeout",
        ("[claude] timeout request_sha256=",),
    ),
)

SNAPSHOT_ROOTS = ("synthesis", "scripts", "tests", "deploy", "grammars", "syncode")
SNAPSHOT_FILES = (
    "pyproject.toml",
    "pytest.ini",
    "setup.cfg",
    "setup.py",
    "requirements.txt",
    "results_matrix.md",
)
ALLOWED_ROOTS = {"synthesis", "scripts", "tests", "deploy"}
SECRET_FILENAMES = {".env", "secrets.json", "credentials.json"}


@dataclasses.dataclass(frozen=True)
class Incident:
    rule: str
    source: str
    line: str
    fingerprint: str


@dataclasses.dataclass(frozen=True)
class RepairResult:
    status: str
    summary: str
    files_changed: list[str]
    tests: list[str]
    safe_to_relaunch: bool


def read_new_text(
    path: Path, cursor: dict[str, int] | None, *, bootstrap_at_end: bool
) -> tuple[str, dict[str, int]]:
    """Read only bytes added since the last scan, surviving truncation/rotation."""
    stat = path.stat()
    if cursor is None:
        offset = stat.st_size if bootstrap_at_end else 0
    elif cursor.get("inode") != stat.st_ino or cursor.get("offset", 0) > stat.st_size:
        offset = 0
    else:
        offset = cursor.get("offset", 0)
    with path.open("rb") as handle:
        handle.seek(offset)
        content = handle.read()
        next_offset = handle.tell()
    return content.decode("utf-8", "replace"), {"inode": stat.st_ino, "offset": next_offset}


def detect_incidents(source: Path, text: str) -> list[Incident]:
    incidents: list[Incident] = []
    for line in text.splitlines():
        lowered = line.lower()
        for rule, phrases in RULES:
            if not any(phrase in lowered for phrase in phrases):
                continue
            if rule == "worker_failed" and ("status=0" in lowered or "exit=0" in lowered):
                continue
            if rule == "worker_failed" and any(
                status in lowered for status in ("status=75", "status=76")
            ):
                continue
            digest = hashlib.sha256(f"{rule}\0{source}\0{line}".encode()).hexdigest()[:20]
            incidents.append(Incident(rule, str(source), line, digest))
            break
    return incidents


def should_escalate_incident(
    incident: Incident,
    state: dict[str, list[float]],
    *,
    now: float | None = None,
    timeout_threshold: int = 3,
    timeout_window_seconds: float = 600.0,
) -> bool:
    """Escalate ordinary incidents immediately and repeated Claude timeouts only."""
    if incident.rule != "claude_timeout":
        return True
    observed = time.time() if now is None else now
    recent = [
        stamp
        for stamp in state.get("claude_timeout", [])
        if observed - stamp <= timeout_window_seconds
    ]
    recent.append(observed)
    state["claude_timeout"] = recent
    return len(recent) >= timeout_threshold


def load_monitor_jobs(path: Path) -> list[dict[str, object]]:
    """Load either the current cold manifest or the older warm manifest."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid monitor manifest {path}: {exc}") from exc
    if isinstance(payload, dict) and isinstance(payload.get("jobs"), list):
        _, jobs = load_cold_manifest(path)
        return jobs
    return load_warm_manifest(path)


def is_protected_path(path: Path) -> bool:
    lowered = "/".join(path.parts).lower()
    name = path.name.lower()
    return (
        name == ".env"
        or name == "results_matrix.md"
        or "grammars/" in f"{lowered}/"
        or lowered.startswith("environment/benchmark_splits/")
        or any(term in lowered for term in ("grader", "scorer", "equivalence"))
    )


def is_allowed_change(path: Path) -> bool:
    if not path.parts or path.parts[0] not in ALLOWED_ROOTS or is_protected_path(path):
        return False
    suffixes = {
        "synthesis": {".py"},
        "scripts": {".py", ".sh"},
        "tests": {".py"},
        "deploy": {".service"},
    }
    return path.suffix in suffixes[path.parts[0]]


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_repair_attestation(
    target: Path,
    *,
    repo: Path,
    base_commit: str,
    changes: dict[Path, str],
    incident_fingerprint: str,
) -> None:
    previous_files: dict[str, str] = {}
    if target.is_file():
        try:
            previous = json.loads(target.read_text(encoding="utf-8"))
            if previous.get("base_commit") == base_commit and isinstance(previous.get("files"), dict):
                previous_files = {str(path): str(digest) for path, digest in previous["files"].items()}
        except (OSError, json.JSONDecodeError):
            previous_files = {}
    candidates = set(previous_files) | {str(relative) for relative, kind in changes.items() if kind != "deleted"}
    files = {
        relative: _file_hash(repo / relative)
        for relative in candidates
        if (repo / relative).is_file()
    }
    payload = {
        "base_commit": base_commit,
        "verifier_exit": 0,
        "incident_fingerprint": incident_fingerprint,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)


def _restore_attestation(path: Path | None, previous: bytes | None) -> None:
    if path is None:
        return
    if previous is None:
        path.unlink(missing_ok=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(previous)


def _tree_files(root: Path) -> dict[Path, str]:
    return {
        path.relative_to(root): _file_hash(path)
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and ".git" not in path.parts
    }


def changed_files(before: Path, after: Path) -> dict[Path, str]:
    old = _tree_files(before)
    new = _tree_files(after)
    return {
        path: "added" if path not in old else "deleted" if path not in new else "modified"
        for path in sorted(set(old) | set(new))
        if old.get(path) != new.get(path)
    }


def repair_environment(source: Mapping[str, str], config_dir: Path) -> dict[str, str]:
    """Strip cloud/API credentials and pin the isolated Claude config directory."""
    blocked_prefixes = ("AWS_", "BEDROCK_")
    clean = {
        key: value
        for key, value in source.items()
        if not key.startswith(blocked_prefixes) and not key.endswith("_API_KEY")
    }
    clean["CLAUDE_CONFIG_DIR"] = str(config_dir)
    return clean


def build_repair_command(
    executable: Path, model: str, result_schema: Path, incident_dir: Path
) -> list[str]:
    """Headless Claude Code call whose final output must match the result schema."""
    return [
        str(executable),
        "-p",
        "--model", model,
        "--output-format", "json",
        "--json-schema", result_schema.read_text(encoding="utf-8"),
        "--permission-mode", "acceptEdits",
        "--allowedTools", "Bash(*)",
        "--add-dir", str(incident_dir),
    ]


def parse_repair_result(stdout_text: str) -> RepairResult:
    """Read the RepairResult from the CLI's JSON envelope; raise ValueError otherwise."""
    envelope = json.loads(stdout_text)
    if envelope.get("is_error") or envelope.get("subtype") != "success":
        raise ValueError(
            f"claude repair run failed subtype={envelope.get('subtype')!r} "
            f"is_error={envelope.get('is_error')!r}"
        )
    payload = envelope.get("structured_output")
    if not isinstance(payload, dict):
        raise ValueError("claude repair envelope has no structured_output object")
    try:
        return RepairResult(**payload)
    except TypeError as error:
        raise ValueError(f"structured_output does not match RepairResult: {error}") from error


def verify_repair_account(executable: Path, config_dir: Path, expected_account: str) -> None:
    """Require the dedicated first-party Max account before any repair call."""
    status_run = subprocess.run(
        [str(executable), "auth", "status", "--json"],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        env=repair_environment(os.environ, config_dir),
        timeout=120,
    )
    if status_run.returncode != 0:
        raise ValueError(f"claude auth status failed with exit {status_run.returncode}")
    status = json.loads(status_run.stdout)
    expected = {
        "loggedIn": True,
        "email": expected_account,
        "authMethod": "claude.ai",
        "apiProvider": "firstParty",
        "subscriptionType": "max",
    }
    mismatches = [
        f"{name}={status.get(name)!r} (expected {value!r})"
        for name, value in expected.items()
        if status.get(name) != value
    ]
    if mismatches:
        raise ValueError(
            "claude repair agent must use the approved Max account: " + "; ".join(mismatches)
        )


def build_repair_prompt(incident: Incident, evidence: Path) -> str:
    return f"""You are repairing one catastrophic operational failure in Dynamic CSD generation.

Incident rule: {incident.rule}
Source log: {incident.source}
Trigger line: {incident.line}
Evidence bundle: {evidence}

Work only inside this isolated source snapshot. First reproduce the defect with a failing test,
then make the smallest general fix and run the focused tests. Add useful diagnostic logging.

Hard boundaries:
- Do not launch synthesis or held-out evaluation.
- Do not call Bedrock, AWS, OpenAI APIs, or any paid provider.
- Do not read or write credentials.
- Do not change grammars, graders, scorers, equivalence rules, dataset splits, datasets,
  model choices, score bars, recipes, the warm/cold policy, results_matrix.md, or the recovery manifest.
- Do not delete files.
- Allowed edits are operational/framework Python, scripts, tests, and focal service files only.

Return the required JSON result. Mark safe_to_relaunch true only after the new failing test and
the relevant existing tests pass. If the incident cannot be safely repaired within these limits,
return status not_repaired and safe_to_relaunch false.
"""


def repair_can_relaunch(
    result: RepairResult,
    *,
    agent_exit: int,
    verifier_exit: int,
    protected_changed: bool,
) -> bool:
    return (
        agent_exit == 0
        and verifier_exit == 0
        and not protected_changed
        and result.status == "repaired"
        and result.safe_to_relaunch
    )


def result_matches_changes(result: RepairResult, changes: dict[Path, str]) -> bool:
    """Require the repair agent's structured claim to match the independently measured edit set."""
    actual = {str(path) for path in changes}
    claimed = set(result.files_changed)
    return bool(actual) and claimed == actual and bool(result.tests)


def _snapshot_ignore(_directory: str, names: list[str]) -> set[str]:
    ignored = {"__pycache__", "outputs", "logs", ".context"}
    return {
        name
        for name in names
        if name in ignored
        or name.endswith(".pyc")
        or name in SECRET_FILENAMES
        or name.startswith(".env.")
    }


def _copy_snapshot(repo: Path, target: Path) -> None:
    for relative in SNAPSHOT_ROOTS:
        source = repo / relative
        if source.exists():
            shutil.copytree(
                source,
                target / relative,
                ignore=_snapshot_ignore,
            )
    split_source = repo / "environment" / "benchmark_splits"
    if split_source.exists():
        shutil.copytree(split_source, target / "environment" / "benchmark_splits")
    for relative in SNAPSHOT_FILES:
        source = repo / relative
        if source.is_file():
            (target / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target / relative)


def _run(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> int:
    LOGGER.warning("[codex-incident] command=%s cwd=%s", command, cwd)
    return subprocess.run(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL).returncode


def _verify(python: Path, repo: Path) -> int:
    changed_python = [str(path) for path in repo.rglob("*.py") if ".git" not in path.parts]
    compile_status = _run([str(python), "-m", "py_compile", *changed_python], cwd=repo)
    if compile_status:
        return compile_status
    return _run(
        [
            str(python), "-m", "pytest", "-q",
            "tests/runtime",
            "tests/test_warm_resume_task_context.py",
            "tests/test_synthesis_run_defaults.py",
        ],
        cwd=repo,
    )


def _capture_evidence(repo: Path, incident: Incident, incident_dir: Path) -> Path:
    incident_dir.mkdir(parents=True, exist_ok=True)
    source = Path(incident.source)
    tail = ""
    if source.is_file():
        tail = "\n".join(source.read_text(encoding="utf-8", errors="replace").splitlines()[-200:])
    commands = {
        "gpu.txt": ["nvidia-smi"],
        "processes.txt": ["ps", "-eo", "pid,ppid,stat,etime,args"],
        "recovery-service.txt": [
            "systemctl", "--user", "status", "csd-cold-synthesis-queue.service", "--no-pager"
        ],
    }
    for filename, command in commands.items():
        result = subprocess.run(command, capture_output=True, text=True, stdin=subprocess.DEVNULL)
        (incident_dir / filename).write_text(result.stdout + result.stderr, encoding="utf-8")
    payload = dataclasses.asdict(incident) | {"captured_at": datetime.now(timezone.utc).isoformat()}
    evidence = incident_dir / "incident.json"
    evidence.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (incident_dir / "source-tail.log").write_text(tail + "\n", encoding="utf-8")
    return evidence


def _stop_recovery(repo: Path, manifest: Path, services: list[str]) -> None:
    for service in services:
        stop_status = _run(["systemctl", "--user", "stop", service])
        if stop_status != 0:
            LOGGER.warning(
                "[codex-incident] service stop returned status=%s service=%s",
                stop_status,
                service,
            )
    jobs = load_monitor_jobs(manifest)
    remaining = recovery_processes(Path("/proc"), jobs, expected_repo=repo)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in remaining:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        if remaining:
            time.sleep(2 if sig == signal.SIGTERM else 1)
            remaining = recovery_processes(Path("/proc"), jobs, expected_repo=repo)
    if remaining:
        raise RuntimeError(f"recovery processes still running after stop: {sorted(remaining)}")
    active = [
        service
        for service in services
        if _run(["systemctl", "--user", "is-active", "--quiet", service]) == 0
    ]
    if active:
        raise RuntimeError(f"recovery services still active after stop: {active}")


def _log_recovery_stopped(services: list[str]) -> None:
    LOGGER.error(
        "[codex-incident] recovery remains stopped services=%s; "
        "manual review is required before relaunch",
        services,
    )


def _confirm_recovery_stopped(
    args: argparse.Namespace, services: list[str]
) -> None:
    try:
        _stop_recovery(args.repo, args.manifest, services)
    except Exception:
        LOGGER.exception(
            "[codex-incident] REPAIR_BLOCKED could not confirm recovery stopped "
            "services=%s",
            services,
        )
    _log_recovery_stopped(services)


def _load_json_state(path: Path, default: object) -> object:
    if not path.is_file():
        return default
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.warning(
            "[codex-incident] ignoring damaged state file path=%s error=%s",
            path,
            exc,
        )
        return default
    if not isinstance(payload, type(default)):
        LOGGER.warning(
            "[codex-incident] ignoring state file with wrong shape path=%s expected=%s",
            path,
            type(default).__name__,
        )
        return default
    return payload


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _process_incidents(
    args: argparse.Namespace,
    incidents: Iterable[Incident],
    seen: set[str],
    timeout_state: dict[str, list[float]],
) -> None:
    for incident in incidents:
        if incident.fingerprint in seen:
            continue
        LOGGER.error(
            "[codex-incident] detected rule=%s source=%s line=%s",
            incident.rule,
            incident.source,
            incident.line,
        )
        if not should_escalate_incident(incident, timeout_state):
            seen.add(incident.fingerprint)
            LOGGER.warning(
                "[codex-incident] deterministic-retry rule=%s fingerprint=%s",
                incident.rule,
                incident.fingerprint,
            )
            continue
        try:
            repaired = repair_incident(args, incident)
        except Exception:
            LOGGER.exception(
                "[codex-incident] REPAIR_BLOCKED unexpected monitor exception"
            )
            services = list(
                getattr(args, "recovery_service", None)
                or ["csd-warm-recovery.service"]
            )
            _confirm_recovery_stopped(args, services)
            repaired = False
        if repaired:
            seen.add(incident.fingerprint)
        LOGGER.warning(
            "[codex-incident] outcome=%s fingerprint=%s",
            "repaired_and_relaunched" if repaired else "REPAIR_BLOCKED",
            incident.fingerprint,
        )


def _deploy_with_rollback(repo: Path, snapshot: Path, changes: dict[Path, str]) -> Path:
    backup = Path(tempfile.mkdtemp(prefix="csd-incident-backup-"))
    for relative, kind in changes.items():
        if kind == "deleted" or not is_allowed_change(relative):
            raise ValueError(f"unsafe change rejected: {kind} {relative}")
    try:
        for relative in changes:
            live = repo / relative
            if live.exists():
                (backup / relative).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(live, backup / relative)
            live.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(snapshot / relative, live)
    except Exception:
        _rollback(repo, backup, changes)
        raise
    return backup


def _rollback(repo: Path, backup: Path, changes: dict[Path, str]) -> None:
    for relative, kind in changes.items():
        saved = backup / relative
        live = repo / relative
        if saved.exists():
            live.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(saved, live)
        elif kind == "added":
            live.unlink(missing_ok=True)


def _clean_up_deployed_repair(
    args: argparse.Namespace,
    services: list[str],
    backup: Path,
    changes: dict[Path, str],
    attestation_path: Path | None,
    previous_attestation: bytes | None,
) -> None:
    _confirm_recovery_stopped(args, services)
    try:
        _rollback(args.repo, backup, changes)
    except Exception:
        LOGGER.exception(
            "[codex-incident] REPAIR_BLOCKED could not roll back deployed repair"
        )
    try:
        _restore_attestation(attestation_path, previous_attestation)
    except Exception:
        LOGGER.exception(
            "[codex-incident] REPAIR_BLOCKED could not restore repair attestation"
        )


def repair_incident(args: argparse.Namespace, incident: Incident) -> bool:
    try:
        verify_repair_account(
            args.claude_executable, args.claude_config_dir, args.claude_expected_account
        )
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        LOGGER.error("[codex-incident] REPAIR_BLOCKED account check failed error=%r", error)
        return False
    attestation_path = getattr(args, "repair_attestation", None)
    previous_attestation = (
        attestation_path.read_bytes() if attestation_path is not None and attestation_path.is_file() else None
    )
    incident_dir = args.state_dir / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{incident.fingerprint}"
    evidence = _capture_evidence(args.repo, incident, incident_dir)
    services = list(getattr(args, "recovery_service", None) or ["csd-warm-recovery.service"])
    _stop_recovery(args.repo, args.manifest, services)
    with tempfile.TemporaryDirectory(prefix="csd-claude-incident-") as temporary:
        baseline = Path(temporary) / "baseline"
        snapshot = Path(temporary) / "workspace"
        _copy_snapshot(args.repo, baseline)
        shutil.copytree(baseline, snapshot)
        prompt_path = incident_dir / "prompt.txt"
        prompt_path.write_text(build_repair_prompt(incident, evidence), encoding="utf-8")
        command = build_repair_command(
            args.claude_executable, args.claude_model, args.result_schema, incident_dir
        )
        LOGGER.warning("[codex-incident] repair-agent command=%s cwd=%s", command, snapshot)
        try:
            agent = subprocess.run(
                command,
                cwd=snapshot,
                env=repair_environment(os.environ, args.claude_config_dir),
                input=prompt_path.read_text(encoding="utf-8"),
                text=True,
                capture_output=True,
                timeout=args.claude_timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            LOGGER.error(
                "[codex-incident] REPAIR_BLOCKED repair agent timed out after %gs",
                args.claude_timeout_seconds,
            )
            _confirm_recovery_stopped(args, services)
            return False
        (incident_dir / "claude-envelope.json").write_text(agent.stdout or "", encoding="utf-8")
        (incident_dir / "claude-stderr.log").write_text(agent.stderr or "", encoding="utf-8")
        changes = changed_files(baseline, snapshot)
        protected_changed = any(is_protected_path(path) for path in changes)
        unsafe_changed = any(kind == "deleted" or not is_allowed_change(path) for path, kind in changes.items())
        try:
            result = parse_repair_result(agent.stdout or "")
        except (ValueError, json.JSONDecodeError) as error:
            LOGGER.error("[codex-incident] invalid repair result error=%r", error)
            _confirm_recovery_stopped(args, services)
            return False
        verifier = _verify(args.python, snapshot) if not unsafe_changed else 1
        if not repair_can_relaunch(
            result,
            agent_exit=agent.returncode,
            verifier_exit=verifier,
            protected_changed=protected_changed or unsafe_changed,
        ) or not result_matches_changes(result, changes):
            LOGGER.error("[codex-incident] REPAIR_BLOCKED result=%s changes=%s", result, changes)
            _confirm_recovery_stopped(args, services)
            return False
        try:
            backup = _deploy_with_rollback(args.repo, snapshot, changes)
        except Exception as error:
            LOGGER.exception("[codex-incident] REPAIR_BLOCKED deployment failed error=%r", error)
            _confirm_recovery_stopped(args, services)
            return False
        try:
            live_verifier = _verify(args.python, args.repo)
        except Exception:
            LOGGER.exception(
                "[codex-incident] REPAIR_BLOCKED live verifier raised"
            )
            _clean_up_deployed_repair(
                args,
                services,
                backup,
                changes,
                attestation_path,
                previous_attestation,
            )
            return False
        if live_verifier:
            LOGGER.error("[codex-incident] REPAIR_BLOCKED live_verifier=%d rollback=%s", live_verifier, backup)
            _clean_up_deployed_repair(
                args,
                services,
                backup,
                changes,
                attestation_path,
                previous_attestation,
            )
            return False
        try:
            manifest_payload = json.loads(args.manifest.read_text(encoding="utf-8"))
            if isinstance(manifest_payload, dict):
                base_commit = str(manifest_payload.get("git_commit", ""))
            else:
                base_commit = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=args.repo,
                    text=True,
                    capture_output=True,
                    check=True,
                ).stdout.strip()
            if attestation_path is not None:
                write_repair_attestation(
                    attestation_path,
                    repo=args.repo,
                    base_commit=base_commit,
                    changes=changes,
                    incident_fingerprint=incident.fingerprint,
                )
        except Exception as error:
            LOGGER.exception(
                "[codex-incident] REPAIR_BLOCKED attestation failed error=%r", error
            )
            _clean_up_deployed_repair(
                args,
                services,
                backup,
                changes,
                attestation_path,
                previous_attestation,
            )
            return False
        LOGGER.warning("[codex-incident] repair verified changes=%s", sorted(map(str, changes)))
    try:
        for service in services:
            start_status = _run(["systemctl", "--user", "start", service])
            active_status = (
                _run(["systemctl", "--user", "is-active", "--quiet", service])
                if start_status == 0
                else start_status
            )
            if start_status or active_status:
                LOGGER.error(
                    "[codex-incident] REPAIR_BLOCKED recovery restart failed "
                    "service=%s start_status=%d active_status=%d",
                    service,
                    start_status,
                    active_status,
                )
                _clean_up_deployed_repair(
                    args,
                    services,
                    backup,
                    changes,
                    attestation_path,
                    previous_attestation,
                )
                return False
    except Exception:
        LOGGER.exception(
            "[codex-incident] REPAIR_BLOCKED recovery restart check raised"
        )
        _clean_up_deployed_repair(
            args,
            services,
            backup,
            changes,
            attestation_path,
            previous_attestation,
        )
        return False
    return True


def _log_paths(manifest: Path, extra_globs: Iterable[str], own_log: Path | None) -> list[Path]:
    jobs = load_monitor_jobs(manifest)
    paths = {Path(str(job["log_file"])) for job in jobs if job.get("log_file")}
    for pattern in extra_globs:
        paths.update(Path().glob(pattern) if not pattern.startswith("/") else Path("/").glob(pattern[1:]))
    if own_log:
        paths.discard(own_log)
    return sorted(path for path in paths if path.is_file())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--claude-executable", type=Path, required=True)
    parser.add_argument("--claude-model", default="claude-sonnet-4-6")
    parser.add_argument("--claude-config-dir", type=Path, required=True)
    parser.add_argument("--claude-expected-account", required=True)
    parser.add_argument("--claude-timeout-seconds", type=float, default=7200)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--result-schema", type=Path, required=True)
    parser.add_argument("--repair-attestation", type=Path)
    parser.add_argument("--extra-log-glob", action="append", default=[])
    parser.add_argument("--own-log", type=Path)
    parser.add_argument("--recovery-service", action="append", default=[])
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
    args.state_dir.mkdir(parents=True, exist_ok=True)
    cursor_path = args.state_dir / "cursors.json"
    seen_path = args.state_dir / "seen.json"
    timeout_path = args.state_dir / "claude_timeout_events.json"
    cursors = dict(_load_json_state(cursor_path, {}))
    seen = set(_load_json_state(seen_path, []))
    timeout_state = dict(_load_json_state(timeout_path, {}))
    while True:
        for path in _log_paths(args.manifest, args.extra_log_glob, args.own_log):
            # Every newly discovered file starts at its current end, including files added
            # after monitor startup. This prevents historical failures from being replayed.
            text, cursor = read_new_text(path, cursors.get(str(path)), bootstrap_at_end=True)
            cursors[str(path)] = cursor
            _process_incidents(args, detect_incidents(path, text), seen, timeout_state)
        _write_json_atomic(cursor_path, cursors)
        _write_json_atomic(seen_path, sorted(seen))
        _write_json_atomic(timeout_path, timeout_state)
        if args.once:
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
