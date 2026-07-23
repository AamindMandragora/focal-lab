"""Production babysitter watch loop (observe + Cursor CLI probe).

Callers: `python -m scripts.runtime.zero_acc_babysitter --watch` / systemd.
API: run_watch_loop(repo, cell_log_paths, poll_seconds) blocks and emits human logs.
Safety: does NOT call CursorCliClient.debug_fix on the live deploy tree (that
checks out repair branches and would interrupt the cold queue). Probes CLI auth
and model default (Grok 4.5); on Acc/memory wakes logs markers and records
blocked_needs_human sidecars for human/follow-up repair.

User instruction: launch-full-queue with CursorCliClient + Grok 4.5 + human logs.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable

from scripts.runtime.zero_acc_babysitter.cloud import (
    DEFAULT_CURSOR_AGENT_MODEL,
    CursorCliClient,
    probe_cursor_cli,
)
from scripts.runtime.zero_acc_babysitter.constants import CellState, PathKind
from scripts.runtime.zero_acc_babysitter.human_logs import HumanLogWriter
from scripts.runtime.zero_acc_babysitter.persistence import (
    CellRecord,
    IncidentRecord,
    IncidentStore,
    make_incident_id,
)
from scripts.runtime.zero_acc_babysitter.watcher import (
    build_watch_event,
    latest_attempt_index,
)

logger = logging.getLogger("zero-acc-babysitter")


def run_watch_loop(
    repo: Path,
    cell_logs: dict[str, Path],
    *,
    poll_seconds: float = 15.0,
    stop_flag: Callable[[], bool] | None = None,
) -> None:
    cell_ids = sorted(cell_logs)
    logs = HumanLogWriter(repo)
    store = IncidentStore(repo)
    ok, note = probe_cursor_cli()
    # Construct client so model/default wiring is live; do not debug_fix here.
    client = CursorCliClient(workspace=repo)
    model = client.model or DEFAULT_CURSOR_AGENT_MODEL
    logs.emit(
        "watcher",
        "WATCHER_START",
        f"cells={len(cell_ids)} model={model} auto_repair=disarmed_live_tree",
    )
    if not ok:
        logs.emit("watcher", "CLI_PROBE_FAIL", note[:500])
        raise RuntimeError(f"Cursor CLI probe failed: {note}")
    logs.emit("watcher", "CLI_PROBE_OK", f"model={model} {note[:300]}")
    logs.emit(
        "watcher",
        "CURSOR_CLI_CLIENT_READY",
        f"class={CursorCliClient.__name__} model={model}",
    )

    for cell_id in cell_ids:
        existing = store.load_cell(cell_id)
        if existing is None:
            store.save_cell(CellRecord(cell_id=cell_id, state=CellState.RUNNING.value))

    seen: dict[str, tuple[int, float | None, bool]] = {}
    stop = stop_flag or (lambda: False)
    while not stop():
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
            event = build_watch_event(text, attempt, finished=True)
            if event.accuracy_pct is None and not event.memory_ops:
                continue
            key = (event.attempt_index, event.accuracy_pct, event.memory_ops)
            if seen.get(cell_id) == key:
                continue
            seen[cell_id] = key
            path_kind = PathKind.MEMORY if event.memory_ops else PathKind.HARNESS
            if event.accuracy_pct is not None and event.accuracy_pct > 0:
                path_kind = (
                    PathKind.TIER_B_HELPER
                    if event.accuracy_pct < 15.0
                    else PathKind.HARNESS
                )
                if event.accuracy_pct >= 15.0:
                    logs.emit(
                        cell_id,
                        "ACC_GE_15_CONTINUE",
                        f"attempt={event.attempt_index} acc={event.accuracy_pct}",
                    )
                    continue
            ts = int(time.time())
            incident_id = make_incident_id(cell_id, event.attempt_index, path_kind, ts)
            record = IncidentRecord(
                incident_id=incident_id,
                cell_id=cell_id,
                attempt_index=event.attempt_index,
                path_kind=path_kind.value,
                trigger_unix_ts=ts,
                phase="observed",
                closed=True,
            )
            store.save_incident(record)
            cell = store.load_cell(cell_id) or CellRecord(
                cell_id=cell_id, state=CellState.RUNNING.value
            )
            cell.state = CellState.BLOCKED_NEEDS_HUMAN.value
            cell.active_incident_id = incident_id
            store.save_cell(cell)
            logs.emit(
                cell_id,
                "WATCH_EVENT",
                f"attempt={event.attempt_index} acc={event.accuracy_pct} mem={event.memory_ops}",
            )
            logs.emit(
                cell_id,
                "WAKE_OBSERVED_NO_AUTO_REPAIR",
                f"id={incident_id} kind={path_kind.value} reason=live_tree_unsafe_for_cli_checkout",
            )
            logs.emit(cell_id, "blocked_needs_human", f"id={incident_id}")
        time.sleep(max(1.0, float(poll_seconds)))
