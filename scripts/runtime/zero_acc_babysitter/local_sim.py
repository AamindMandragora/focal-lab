"""Local state-machine simulator — never authorizes a real queue.

Callers: tests/test_zero_acc_babysitter_local_sim.py (all 11 scenarios);
         scratchpad/run_focal_mock_suite.py; __main__.py.
API: LocalBabysitterSim.finish_attempt / abort_memory / resume_from_disk;
     optional repair_client (defaults to NullCloudClient).
Data: uses logs/zero_acc_babysitter via Orchestrator (synthetic tmp_path only).
User instruction: Cursor CLI babysitter; keep local sim working.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from scripts.runtime.zero_acc_babysitter.cloud import NullCloudClient, RepairAgentClient
from scripts.runtime.zero_acc_babysitter.constants import CellState
from scripts.runtime.zero_acc_babysitter.orchestrator import (
    BabysitterHooks,
    HandleResult,
    Orchestrator,
)
from scripts.runtime.zero_acc_babysitter.persistence import IncidentRecord


@dataclass
class LocalBabysitterSim:
    repo_root: Path
    cell_ids: list[str]
    twin_accuracy: float = 0.0
    helper_failures: list[str] = field(default_factory=list)
    helper_trace_by_attempt: dict[int, list[str]] | None = None
    smoke_results: list[bool] | None = None
    crash_after_cloud_attempts: int | None = None
    repair_client: RepairAgentClient | None = None
    merge_count: int = 0
    _smoke_queue: deque[bool] = field(default_factory=deque)
    _orch: Orchestrator = field(init=False)
    _repair: RepairAgentClient = field(init=False)

    def __post_init__(self) -> None:
        results = self.smoke_results if self.smoke_results is not None else [True]
        self._smoke_queue = deque(results)
        self._repair = self.repair_client or NullCloudClient()
        hooks = BabysitterHooks(
            twin_accuracy=lambda _c: float(self.twin_accuracy),
            helper_failures=lambda _c: list(self.helper_failures),
            run_cloud_debug=self._cloud,
            run_smoke=self._smoke,
            merge_and_pull=self._merge,
            kill_cell=lambda _c: None,
            broken_sha=lambda: "brokendeadbeef",
            rescore_all=lambda _c, _s: None,
            sibling_reeval=lambda _c, _s, _sibs: None,
            restart_from_k=lambda _c, _k: None,
            memory_resume=lambda _c, _a: None,
        )
        self._orch = Orchestrator(
            repo_root=self.repo_root,
            cell_ids=list(self.cell_ids),
            hooks=hooks,
            crash_after_cloud_attempts=self.crash_after_cloud_attempts,
        )
        if self.helper_trace_by_attempt:
            for cell_id in self.cell_ids:
                self._orch.set_helper_traces(cell_id, self.helper_trace_by_attempt)

    @classmethod
    def resume_from_disk(
        cls,
        *,
        repo_root: Path,
        cell_ids: list[str],
        twin_accuracy: float = 0.0,
        helper_failures: list[str] | None = None,
        smoke_results: list[bool] | None = None,
        repair_client: RepairAgentClient | None = None,
    ) -> "LocalBabysitterSim":
        return cls(
            repo_root=repo_root,
            cell_ids=cell_ids,
            twin_accuracy=twin_accuracy,
            helper_failures=list(helper_failures or []),
            smoke_results=smoke_results,
            crash_after_cloud_attempts=None,
            repair_client=repair_client,
        )

    def _cloud(self, incident: IncidentRecord) -> str | None:
        return self._repair.debug_fix(incident)

    def _smoke(self, _incident: IncidentRecord) -> bool:
        if not self._smoke_queue:
            return True
        return bool(self._smoke_queue.popleft())

    def _merge(self, _incident: IncidentRecord) -> str:
        self.merge_count += 1
        return "mergedcafebabe"

    def finish_attempt(
        self, cell_id: str, *, attempt_index: int, accuracy_pct: float | None
    ) -> HandleResult:
        return self._orch.handle_event(
            cell_id,
            attempt_index,
            memory_ops=False,
            finished=True,
            accuracy_pct=accuracy_pct,
        )

    def abort_memory(self, cell_id: str, *, attempt_index: int) -> HandleResult:
        return self._orch.handle_event(
            cell_id,
            attempt_index,
            memory_ops=True,
            finished=False,
            accuracy_pct=None,
        )

    def abort_memory_then_finished(
        self, cell_id: str, *, attempt_index: int, accuracy_pct: float | None
    ) -> HandleResult:
        return self._orch.handle_event(
            cell_id,
            attempt_index,
            memory_ops=True,
            finished=True,
            accuracy_pct=accuracy_pct,
            then_accuracy_after_memory=True,
        )

    def resume_active_incident(self, cell_id: str) -> HandleResult:
        return self._orch.resume_active_incident(cell_id)

    def master_text(self) -> str:
        return self._orch.logs.read_master()

    def lane_text(self, lane: str) -> str:
        return self._orch.logs.read_lane(lane)

    def cell_state(self, cell_id: str) -> CellState:
        return CellState(self._orch.cells[cell_id].state)

    def was_killed(self, cell_id: str) -> bool:
        return self._orch.cells[cell_id].kill_count > 0

    def kill_count(self, cell_id: str) -> int:
        return self._orch.cells[cell_id].kill_count

    def restart_k(self, cell_id: str) -> int | None:
        return self._orch.cells[cell_id].restart_k

    def cloud_attempt_count(self, cell_id: str) -> int:
        cell = self._orch.cells[cell_id]
        if cell.active_incident_id:
            incident = self._orch.store.load_incident(cell.active_incident_id)
            if incident:
                return incident.cloud_attempt_count
        best = 0
        for path in self._orch.store.incidents_dir.glob("*.json"):
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("cell_id") == cell_id:
                best = max(best, int(data.get("cloud_attempt_count", 0)))
        return best

    def active_incident_id(self, cell_id: str) -> str | None:
        return self._orch.cells[cell_id].active_incident_id
