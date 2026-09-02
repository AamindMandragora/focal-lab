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


def _real_orchestrator(live: Path, cell: str = CELL):
    """Real Orchestrator with recording stub hooks (no processes, no git)."""
    from scripts.runtime.zero_acc_babysitter.orchestrator import (
        BabysitterHooks,
        Orchestrator,
    )

    calls: dict[str, list] = {"kill": [], "memory_resume": []}
    hooks = BabysitterHooks(
        twin_accuracy=lambda _c: 0.0,
        helper_failures=lambda _c: [],
        run_cloud_debug=lambda _i: "https://example.invalid/pr/1",
        run_smoke=lambda _i: True,
        merge_and_pull=lambda _i: "deadbeef",
        kill_cell=lambda c: calls["kill"].append(c),
        broken_sha=lambda: "cafebabe",
        rescore_all=lambda _c, _s: None,
        sibling_reeval=lambda _c, _s, _sib: None,
        restart_from_k=lambda _c, _k: None,
        memory_resume=lambda c, k: calls["memory_resume"].append((c, k)),
    )
    orch = Orchestrator(repo_root=live, cell_ids=[cell], hooks=hooks)
    return orch, calls


def test_orchestrator_skips_stale_memory_for_closed_attempt(tmp_path: Path) -> None:
    """Regression for smiles-acrylates-qwen35-2b:2:memory:1784887455: a watcher
    process running pre-guard code re-delivered attempt 2's stale OOM marker
    seconds after the incident closed; handle_memory must refuse to re-open."""
    live = tmp_path / "live"
    live.mkdir()
    orch, calls = _real_orchestrator(live)
    _closed_memory_incident(orch.store, attempt=3, ts=1784879589)

    markers: list[str] = []
    orch.logs.emit = lambda _c, marker, _d="": markers.append(marker)

    result = orch.handle_memory(CELL, 3)

    assert not result.woke
    assert calls["kill"] == []
    assert "STALE_MEMORY_EVENT_SKIP" in markers
    assert "INCIDENT_OPEN" not in markers
    assert orch.store.list_open_incidents() == []


def test_orchestrator_fresh_memory_attempt_still_handled(tmp_path: Path) -> None:
    live = tmp_path / "live"
    live.mkdir()
    orch, calls = _real_orchestrator(live)
    # Closed incident on attempt 3 must not suppress a genuine OOM on attempt 4.
    _closed_memory_incident(orch.store, attempt=3, ts=1784879589)

    result = orch.handle_memory(CELL, 4)

    assert result.woke
    assert calls["kill"] == [CELL]
    assert calls["memory_resume"] == [(CELL, 4)]


def test_orchestrator_open_incident_not_skipped_as_stale(tmp_path: Path) -> None:
    """A still-open incident for the same (cell, attempt) must attach/resume,
    even when an earlier memory incident for that attempt already closed."""
    live = tmp_path / "live"
    live.mkdir()
    orch, calls = _real_orchestrator(live)
    _closed_memory_incident(orch.store, attempt=3, ts=1784879589)
    open_id = f"{CELL}:3:memory:1784887455"
    orch.store.save_incident(
        IncidentRecord(
            incident_id=open_id,
            cell_id=CELL,
            attempt_index=3,
            path_kind=PathKind.MEMORY.value,
            trigger_unix_ts=1784887455,
            closed=False,
        )
    )
    orch.cells[CELL].active_incident_id = open_id
    orch.store.save_cell(orch.cells[CELL])

    result = orch.handle_memory(CELL, 3)

    assert result.woke
    assert calls["memory_resume"] == [(CELL, 3)]
    closed = orch.store.load_incident(open_id)
    assert closed is not None and closed.closed


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
