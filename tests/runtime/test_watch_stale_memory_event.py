"""Stale memory events must not re-open incidents after resume/restart.

Regression for spider-qwen25-1p5b:3:memory:1784878709: after a memory
incident closed (MEMORY_OPS_RESUME), the cell log still contained attempt
3's OOM marker; a watcher restart emptied the in-memory `seen` map and the
next poll re-opened a duplicate memory incident, looping forever.
"""

from __future__ import annotations

from pathlib import Path

from scripts.runtime.zero_acc_babysitter.constants import PathKind
from scripts.runtime.zero_acc_babysitter.orchestrator import HandleResult
from scripts.runtime.zero_acc_babysitter.persistence import (
    IncidentRecord,
    IncidentStore,
)
from scripts.runtime.zero_acc_babysitter.production_watch import run_watch_once

CELL = "spider-qwen25-1p5b"

STALE_LOG = """Attempt 3/30
torch.OutOfMemoryError: CUDA out of memory
Accuracy: 0.0%
COLDQ_SYNTHESIS_FINISH
"""


class _FakeOrch:
    def __init__(self, live: Path) -> None:
        from scripts.runtime.zero_acc_babysitter.human_logs import HumanLogWriter

        self.store = IncidentStore(live)
        self.logs = HumanLogWriter(live)
        self.handled: list[tuple[str, int, bool]] = []

    def resume_active_incident(self, cell_id: str) -> HandleResult:
        return HandleResult(woke=False)

    def handle_event(self, cell_id, attempt_index, **kwargs) -> HandleResult:
        self.handled.append((cell_id, attempt_index, kwargs.get("memory_ops", False)))
        return HandleResult(woke=True)


def _setup(tmp_path: Path) -> tuple[Path, Path, _FakeOrch]:
    live = tmp_path / "live"
    live.mkdir()
    log_path = live / "cell.log"
    log_path.write_text(STALE_LOG, encoding="utf-8")
    return live, log_path, _FakeOrch(live)


def _closed_memory_incident(store: IncidentStore, attempt: int, ts: int) -> None:
    store.save_incident(
        IncidentRecord(
            incident_id=f"{CELL}:{attempt}:memory:{ts}",
            cell_id=CELL,
            attempt_index=attempt,
            path_kind=PathKind.MEMORY.value,
            trigger_unix_ts=ts,
            closed=True,
        )
    )


def test_stale_memory_event_skipped_after_incident_close(tmp_path: Path) -> None:
    live, log_path, orch = _setup(tmp_path)
    _closed_memory_incident(orch.store, attempt=3, ts=1784875953)

    markers: list[str] = []
    seen: dict = {}
    woke = run_watch_once(
        live_repo=live,
        cell_logs={CELL: log_path},
        seen=seen,
        repair_worktree=tmp_path / "repair",
        auto_repair=True,
        emit=lambda _c, marker, _d="": markers.append(marker),
        orch=orch,
    )

    assert not woke
    assert orch.handled == []
    assert "STALE_MEMORY_EVENT_SKIP" in markers
    # Marked seen so subsequent ticks stay quiet too.
    assert seen[CELL] == (3, 0.0, True)


def test_fresh_memory_event_still_wakes(tmp_path: Path) -> None:
    live, log_path, orch = _setup(tmp_path)
    # Closed incident for a different attempt must not suppress attempt 3.
    _closed_memory_incident(orch.store, attempt=2, ts=1784800000)

    woke = run_watch_once(
        live_repo=live,
        cell_logs={CELL: log_path},
        seen={},
        repair_worktree=tmp_path / "repair",
        auto_repair=True,
        emit=lambda _c, _m, _d="": None,
        orch=orch,
    )

    assert woke
    assert orch.handled == [(CELL, 3, True)]
