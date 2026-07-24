"""Restarted runs appended to one log must not mark in-progress attempts done.

Regression for spider-qwen25-1p5b:2:telemetry:1784888533: the cell log held
four appended synthesis runs, each restarting attempt numbering. The latest
"Attempt 2" was still running, but is_attempt_finished bound to the FIRST
"Attempt 2" (an old run followed by later attempt headers) and declared it
finished; its block had no Accuracy line yet, opening a false telemetry
incident and killing the live run.
"""

from __future__ import annotations

from pathlib import Path

from scripts.runtime.zero_acc_babysitter.production_watch import run_watch_once
from scripts.runtime.zero_acc_babysitter.watcher import is_attempt_finished

CELL = "spider-qwen25-1p5b"

RESTARTED_LOG = """Attempt 1/40
    Accuracy: 1.3% (min: 53.3%)
Attempt 2/40
COLDQ_SYNTHESIS_FINISH cell=spider-qwen25-1p5b status=-15
Attempt 1/40
    Accuracy: 10.3% (min: 53.3%)
Attempt 2/40
"""


def test_in_progress_restarted_attempt_not_finished() -> None:
    assert not is_attempt_finished(RESTARTED_LOG, 2, pid_file=None)


def test_restarted_attempt_finished_after_finish_marker() -> None:
    text = RESTARTED_LOG + "COLDQ_SYNTHESIS_FINISH cell=spider-qwen25-1p5b status=0\n"
    assert is_attempt_finished(text, 2, pid_file=None)


def test_watch_does_not_wake_on_in_progress_restarted_attempt(tmp_path: Path) -> None:
    from scripts.runtime.zero_acc_babysitter.human_logs import HumanLogWriter
    from scripts.runtime.zero_acc_babysitter.orchestrator import HandleResult
    from scripts.runtime.zero_acc_babysitter.persistence import IncidentStore

    live = tmp_path / "live"
    live.mkdir()
    log_path = live / "cell.log"
    log_path.write_text(RESTARTED_LOG, encoding="utf-8")

    class _FakeOrch:
        def __init__(self) -> None:
            self.store = IncidentStore(live)
            self.logs = HumanLogWriter(live)
            self.handled: list[tuple[str, int]] = []

        def resume_active_incident(self, cell_id: str) -> HandleResult:
            return HandleResult(woke=False)

        def handle_event(self, cell_id, attempt_index, **kwargs) -> HandleResult:
            self.handled.append((cell_id, attempt_index))
            return HandleResult(woke=True)

    orch = _FakeOrch()
    woke = run_watch_once(
        live_repo=live,
        cell_logs={CELL: log_path},
        seen={},
        repair_worktree=tmp_path / "repair",
        auto_repair=True,
        emit=lambda _c, _m, _d="": None,
        orch=orch,
    )

    assert not woke
    assert orch.handled == []
