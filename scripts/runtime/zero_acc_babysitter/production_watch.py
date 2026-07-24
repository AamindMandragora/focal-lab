"""Production babysitter watch loop wired through Orchestrator → merge.

Callers: `python -m scripts.runtime.zero_acc_babysitter --watch` / systemd.
API: run_watch_loop(...); handle_watch_wake(...); run_watch_once(...).
Safety: ClaudeCodeCliClient.debug_fix runs only in a sibling git worktree so
`git checkout -B babysitter-fix/...` never moves the live cold-queue tree.
After smoke PASS, hooks.merge_and_pull merges into synthesis-snapshot-20260622
and pulls on the live tree (Orchestrator logs MERGED sha=...).

User instruction: wire production_watch end-to-end through merge.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable

from scripts.runtime.zero_acc_babysitter.cloud import (
    DEFAULT_CLAUDE_CODE_MODEL,
    ClaudeCodeCliClient,
    NullCloudClient,
    probe_claude_code_cli,
)
from scripts.runtime.zero_acc_babysitter.constants import CellState, TIER_B_ACC_PCT
from scripts.runtime.zero_acc_babysitter.human_logs import HumanLogWriter
from scripts.runtime.zero_acc_babysitter.orchestrator import (
    BabysitterHooks,
    HandleResult,
    Orchestrator,
)
from scripts.runtime.zero_acc_babysitter.persistence import CellRecord, IncidentStore
from scripts.runtime.zero_acc_babysitter.production_hooks import build_production_hooks
from scripts.runtime.zero_acc_babysitter.repair_worktree import (
    default_repair_worktree_path,
    ensure_repair_worktree,
    live_head_sha,
)
from scripts.runtime.zero_acc_babysitter.watcher import (
    build_watch_event,
    is_attempt_finished,
    latest_attempt_index,
)

logger = logging.getLogger("zero-acc-babysitter")

EmitFn = Callable[[str, str, str], None]
SeenMap = dict[str, tuple[int, float | None, bool]]


def _bridge_emit(logs: HumanLogWriter, emit: EmitFn | None) -> EmitFn:
    def _emit(cell_id: str, marker: str, detail: str = "") -> None:
        logs.emit(cell_id, marker, detail)
        if emit is not None:
            emit(cell_id, marker, detail)

    return _emit


def build_watch_orchestrator(
    *,
    live_repo: Path,
    repair_worktree: Path,
    cell_ids: list[str],
    auto_repair: bool,
    hooks: BabysitterHooks | None = None,
    max_cloud_attempts: int | None = None,
    client: ClaudeCodeCliClient | None = None,
) -> Orchestrator:
    live = live_repo.resolve()
    repair = repair_worktree.resolve()
    if hooks is None:
        if auto_repair:
            repair_client = client or ClaudeCodeCliClient(
                workspace=repair,
                base_ref=live_head_sha(live),
            )
        else:
            repair_client = NullCloudClient()
        hooks = build_production_hooks(
            live_repo=live,
            repair_worktree=repair,
            repair_client=repair_client,
            cell_ids=cell_ids,
        )
    orch = Orchestrator(repo_root=live, cell_ids=list(cell_ids), hooks=hooks)
    if max_cloud_attempts is not None:
        orch.max_cloud_attempts = int(max_cloud_attempts)
    return orch


def handle_watch_wake(
    *,
    live_repo: Path,
    cell_id: str,
    attempt_index: int,
    accuracy_pct: float | None,
    memory_ops: bool,
    repair_worktree: Path,
    auto_repair: bool,
    emit: EmitFn,
    store: IncidentStore | None = None,
    hooks: BabysitterHooks | None = None,
    cell_ids: list[str] | None = None,
    orch: Orchestrator | None = None,
    client: ClaudeCodeCliClient | None = None,
    max_cloud_attempts: int | None = None,
) -> HandleResult:
    """Run one wake through Orchestrator (kill → Cloud → smoke → merge → recovery)."""
    live = live_repo.resolve()
    repair = repair_worktree.resolve()
    if repair == live:
        raise ValueError("repair worktree path must differ from live repo")

    cells = cell_ids or [cell_id]
    if orch is None:
        # When hooks carry a max override for tests, apply via kwargs.
        override = None
        if hooks is not None and max_cloud_attempts is None:
            # production_hooks may stash override on a sentinel attr
            override = getattr(hooks, "_max_cloud_attempts_override", None)
        orch = build_watch_orchestrator(
            live_repo=live,
            repair_worktree=repair,
            cell_ids=cells,
            auto_repair=auto_repair,
            hooks=hooks,
            max_cloud_attempts=max_cloud_attempts if max_cloud_attempts is not None else override,
            client=client,
        )
    bridged = _bridge_emit(orch.logs, emit)
    store = store or orch.store

    bridged(
        cell_id,
        "WATCH_EVENT",
        f"attempt={attempt_index} acc={accuracy_pct} mem={memory_ops}",
    )

    if not auto_repair:
        cell = store.load_cell(cell_id) or CellRecord(
            cell_id=cell_id, state=CellState.RUNNING.value
        )
        cell.state = CellState.BLOCKED_NEEDS_HUMAN.value
        store.save_cell(cell)
        bridged(
            cell_id,
            "WAKE_OBSERVED_NO_AUTO_REPAIR",
            f"attempt={attempt_index} reason=auto_repair_disabled",
        )
        bridged(cell_id, "blocked_needs_human", f"attempt={attempt_index}")
        return HandleResult(woke=True, blocked=True)

    if client is not None:
        client.workspace = repair
        client.base_ref = live_head_sha(live)

    bridged(
        cell_id,
        "WAKE_AUTO_REPAIR_VIA_WORKTREE",
        f"attempt={attempt_index} worktree={repair}",
    )

    # ≤1 active incident: resume rather than opening a parallel Cloud agent.
    existing = store.load_cell(cell_id)
    if existing and existing.active_incident_id:
        open_inc = store.load_incident(existing.active_incident_id)
        if open_inc and not open_inc.closed:
            bridged(
                cell_id,
                "INCIDENT_ATTACH",
                f"id={open_inc.incident_id} cloud_attempts={open_inc.cloud_attempt_count}",
            )
            return orch.resume_active_incident(cell_id)

    try:
        if memory_ops:
            # MEMORY first; chain Acc/TELEMETRY only when Acc present on this wake.
            return orch.handle_event(
                cell_id,
                attempt_index,
                memory_ops=True,
                finished=True,
                accuracy_pct=accuracy_pct,
                then_accuracy_after_memory=accuracy_pct is not None,
            )
        return orch.handle_event(
            cell_id,
            attempt_index,
            memory_ops=False,
            finished=True,
            accuracy_pct=accuracy_pct,
        )
    except Exception as exc:  # noqa: BLE001 — surface as blocked, keep queue alive
        logger.exception("auto repair failed cell=%s", cell_id)
        cell = store.load_cell(cell_id) or CellRecord(
            cell_id=cell_id, state=CellState.RUNNING.value
        )
        cell.state = CellState.BLOCKED_NEEDS_HUMAN.value
        store.save_cell(cell)
        bridged(
            cell_id,
            "CLI_AGENT_FAIL",
            f"err={type(exc).__name__}: {exc}"[:500],
        )
        bridged(cell_id, "blocked_needs_human", f"attempt={attempt_index}")
        return HandleResult(woke=True, blocked=True)


def run_watch_once(
    *,
    live_repo: Path,
    cell_logs: dict[str, Path],
    seen: SeenMap,
    repair_worktree: Path,
    auto_repair: bool,
    emit: EmitFn,
    hooks: BabysitterHooks | None = None,
    cell_ids: list[str] | None = None,
    orch: Orchestrator | None = None,
) -> bool:
    """Process one poll tick. Returns True if any wake ran the fix path."""
    any_wake = False
    ids = cell_ids or sorted(cell_logs)
    for cell_id, log_path in cell_logs.items():
        if not log_path.is_file():
            continue
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.warning("read fail cell=%s err=%s", cell_id, exc)
            continue
        attempt = latest_attempt_index(text)
        if attempt is None:
            continue
        # B1: do not treat an in-progress attempt as finished.
        pid_file = (
            live_repo
            / "logs"
            / "zero_acc_babysitter"
            / "cells"
            / f"{cell_id}.pid"
        )
        finished = is_attempt_finished(text, attempt, pid_file=pid_file)
        if not finished:
            continue
        event = build_watch_event(text, attempt, finished=True)
        # Missing Acc after finished attempt is TELEMETRY_FAIL — do not skip.
        key = (event.attempt_index, event.accuracy_pct, event.memory_ops)
        if seen.get(cell_id) == key:
            continue
        seen[cell_id] = key

        if (
            event.accuracy_pct is not None
            and event.accuracy_pct >= TIER_B_ACC_PCT
            and not event.memory_ops
        ):
            emit(
                cell_id,
                "ACC_GE_15_CONTINUE",
                f"attempt={event.attempt_index} acc={event.accuracy_pct}",
            )
            continue

        result = handle_watch_wake(
            live_repo=live_repo,
            cell_id=cell_id,
            attempt_index=event.attempt_index,
            accuracy_pct=event.accuracy_pct,
            memory_ops=event.memory_ops,
            repair_worktree=repair_worktree,
            auto_repair=auto_repair,
            emit=emit,
            hooks=hooks,
            cell_ids=ids,
            orch=orch,
        )
        if result.woke:
            any_wake = True
    return any_wake


def run_watch_loop(
    repo: Path,
    cell_logs: dict[str, Path],
    *,
    poll_seconds: float = 15.0,
    stop_flag: Callable[[], bool] | None = None,
    repair_worktree: Path | None = None,
    auto_repair: bool = True,
) -> None:
    cell_ids = sorted(cell_logs)
    live = repo.resolve()
    logs = HumanLogWriter(live)
    store = IncidentStore(live)

    repair_path = (repair_worktree or default_repair_worktree_path(live)).resolve()
    if auto_repair:
        ok, note = probe_claude_code_cli()
        ensure_repair_worktree(live, repair_path)
        client: ClaudeCodeCliClient | None = ClaudeCodeCliClient(
            workspace=repair_path,
            base_ref=live_head_sha(live),
            log_emit=lambda cell, marker, detail="": logs.emit(cell, marker, detail),
        )
        mode = f"auto_repair=orchestrator_merge path={repair_path}"
    else:
        ok, note = True, "auto_repair disarmed"
        client = None
        mode = "auto_repair=disarmed"

    orch = build_watch_orchestrator(
        live_repo=live,
        repair_worktree=repair_path,
        cell_ids=cell_ids,
        auto_repair=auto_repair,
        client=client,
    )
    if client is not None:
        client.log_emit = lambda cell, marker, detail="": logs.emit(cell, marker, detail)

    model = getattr(client, "model", None) or DEFAULT_CLAUDE_CODE_MODEL
    logs.emit(
        "watcher",
        "WATCHER_START",
        f"cells={len(cell_ids)} model={model} {mode}",
    )
    if auto_repair and not ok:
        logs.emit("watcher", "CLI_PROBE_FAIL", note[:500])
        raise RuntimeError(f"Claude Code CLI probe failed: {note}")
    if auto_repair:
        logs.emit("watcher", "CLI_PROBE_OK", f"model={model} {note[:300]}")
        logs.emit(
            "watcher",
            "CLAUDE_CODE_CLI_READY",
            f"class={type(client).__name__} model={model} workspace={repair_path}",
        )

    for cell_id in cell_ids:
        existing = store.load_cell(cell_id)
        if existing is None:
            store.save_cell(CellRecord(cell_id=cell_id, state=CellState.RUNNING.value))

    # Mid-incident resume on startup (do not reset cloud_attempt_count).
    for cell_id in cell_ids:
        cell = store.load_cell(cell_id)
        if not cell or not cell.active_incident_id:
            continue
        incident = store.load_incident(cell.active_incident_id)
        if incident is None or incident.closed:
            continue
        logs.emit(
            cell_id,
            "INCIDENT_RESUME_STARTUP",
            f"id={incident.incident_id} cloud_attempts={incident.cloud_attempt_count}",
        )
        try:
            orch.resume_active_incident(cell_id)
        except Exception as exc:  # noqa: BLE001
            logger.exception("startup resume failed cell=%s", cell_id)
            logs.emit(
                cell_id,
                "INCIDENT_RESUME_FAIL",
                f"err={type(exc).__name__}: {exc}"[:400],
            )

    seen: SeenMap = {}
    stop = stop_flag or (lambda: False)
    emit: EmitFn = lambda c, m, d="": logs.emit(c, m, d)
    while not stop():
        run_watch_once(
            live_repo=live,
            cell_logs=cell_logs,
            seen=seen,
            repair_worktree=repair_path,
            auto_repair=auto_repair,
            emit=emit,
            orch=orch,
            cell_ids=cell_ids,
        )
        time.sleep(max(1.0, float(poll_seconds)))
