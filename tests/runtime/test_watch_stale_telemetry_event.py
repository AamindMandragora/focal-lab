"""Stale telemetry events must not re-open incidents after close.

Regression for smiles-chain_extenders-qwen25-1p5b:1:telemetry:1784887767:
telemetry recovery (rescore_all) does not append an Accuracy line to the
cell run log, so after the incident closed the log still showed a finished
attempt with no Accuracy. The next poll (with an empty `seen` map after
watcher restart) re-opened a duplicate telemetry incident.
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

CELL = "smiles-chain_extenders-qwen25-1p5b"

STALE_LOG = """Attempt 1/40
Total attempts: 1
COLDQ_SYNTHESIS_FINISH cell=smiles-chain_extenders-qwen25-1p5b status=0
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


def _closed_telemetry_incident(store: IncidentStore, attempt: int, ts: int) -> None:
    store.save_incident(
        IncidentRecord(
            incident_id=f"{CELL}:{attempt}:telemetry:{ts}",
            cell_id=CELL,
            attempt_index=attempt,
            path_kind=PathKind.TELEMETRY.value,
            trigger_unix_ts=ts,
            closed=True,
        )
    )


def test_stale_telemetry_event_skipped_after_incident_close(tmp_path: Path) -> None:
    live, log_path, orch = _setup(tmp_path)
    _closed_telemetry_incident(orch.store, attempt=1, ts=1784879977)

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
    assert "STALE_TELEMETRY_EVENT_SKIP" in markers
    assert seen[CELL] == (1, None, False)


def _real_orchestrator(live: Path, cell: str = CELL):
    """Real Orchestrator with recording stub hooks (no processes, no git)."""
    from scripts.runtime.zero_acc_babysitter.orchestrator import (
        BabysitterHooks,
        Orchestrator,
    )

    calls: dict[str, list] = {"kill": [], "rescore_all": []}
    hooks = BabysitterHooks(
        twin_accuracy=lambda _c: 0.0,
        helper_failures=lambda _c: [],
        run_cloud_debug=lambda _i: "https://example.invalid/pr/1",
        run_smoke=lambda _i: True,
        merge_and_pull=lambda _i: "deadbeef",
        kill_cell=lambda c: calls["kill"].append(c),
        broken_sha=lambda: "cafebabe",
        rescore_all=lambda c, s: calls["rescore_all"].append((c, s)),
        sibling_reeval=lambda _c, _s, _sib: None,
        restart_from_k=lambda _c, _k: None,
        memory_resume=lambda _c, _k: None,
    )
    orch = Orchestrator(repo_root=live, cell_ids=[cell], hooks=hooks)
    return orch, calls


def test_orchestrator_skips_stale_telemetry_for_closed_attempt(tmp_path: Path) -> None:
    live = tmp_path / "live"
    live.mkdir()
    orch, calls = _real_orchestrator(live)
    _closed_telemetry_incident(orch.store, attempt=1, ts=1784879977)

    markers: list[str] = []
    orch.logs.emit = lambda _c, marker, _d="": markers.append(marker)

    result = orch.handle_accuracy(CELL, 1, None, finished=True)

    assert not result.woke
    assert calls["kill"] == []
    assert "STALE_TELEMETRY_EVENT_SKIP" in markers
    assert "TELEMETRY_FAIL" not in markers
    assert "INCIDENT_OPEN" not in markers
    assert orch.store.list_open_incidents() == []


def test_orchestrator_fresh_telemetry_attempt_still_handled(tmp_path: Path) -> None:
    live = tmp_path / "live"
    live.mkdir()
    orch, calls = _real_orchestrator(live)
    # Closed incident on attempt 1 must not suppress a genuine fail on attempt 2.
    _closed_telemetry_incident(orch.store, attempt=1, ts=1784879977)

    result = orch.handle_accuracy(CELL, 2, None, finished=True)

    assert result.woke
    assert calls["kill"] == [CELL]
    assert calls["rescore_all"] == [(CELL, "cafebabe")]


def test_orchestrator_open_incident_not_skipped_as_stale(tmp_path: Path) -> None:
    """A still-open incident for the same (cell, attempt) must not be treated
    as stale even when an earlier telemetry incident for that attempt closed."""
    live = tmp_path / "live"
    live.mkdir()
    orch, calls = _real_orchestrator(live)
    _closed_telemetry_incident(orch.store, attempt=1, ts=1784879977)
    open_id = f"{CELL}:1:telemetry:1784887767"
    orch.store.save_incident(
        IncidentRecord(
            incident_id=open_id,
            cell_id=CELL,
            attempt_index=1,
            path_kind=PathKind.TELEMETRY.value,
            trigger_unix_ts=1784887767,
            closed=False,
        )
    )
    orch.cells[CELL].active_incident_id = open_id
    orch.store.save_cell(orch.cells[CELL])

    result = orch.handle_accuracy(CELL, 1, None, finished=True)

    assert result.woke


def test_telemetry_event_with_accuracy_not_skipped(tmp_path: Path) -> None:
    """Acc present on the wake means it is not a telemetry fail — the closed
    telemetry incident must not suppress the Acc-tier path."""
    live, log_path, orch = _setup(tmp_path)
    log_path.write_text(
        STALE_LOG.replace("Total attempts: 1", "Accuracy: 0.0%"), encoding="utf-8"
    )
    _closed_telemetry_incident(orch.store, attempt=1, ts=1784879977)

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
    assert orch.handled == [(CELL, 1, False)]
