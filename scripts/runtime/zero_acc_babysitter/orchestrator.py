"""Core babysitter orchestration (memory → accuracy / TELEMETRY, Cloud, smoke, merge).

Callers: local_sim.LocalBabysitterSim, future watcher CLI.
API: Orchestrator.handle_event / resume_active_incident; BabysitterHooks for side effects.
User instruction: babysitter-implement + local-mocks TDD.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from scripts.runtime.zero_acc_babysitter.constants import (
    MAX_CLOUD_ATTEMPTS,
    TIER_B_ACC_PCT,
    CellState,
    IncidentPhase,
    PathKind,
)
from scripts.runtime.zero_acc_babysitter.domains import same_domain_siblings
from scripts.runtime.zero_acc_babysitter.human_logs import HumanLogWriter
from scripts.runtime.zero_acc_babysitter.persistence import (
    CellRecord,
    IncidentRecord,
    IncidentStore,
    make_incident_id,
)


@dataclass
class HandleResult:
    woke: bool = False
    path_kind: PathKind | None = None
    blocked: bool = False
    memory_then_accuracy: bool = False


@dataclass
class BabysitterHooks:
    twin_accuracy: Callable[[str], float]
    helper_failures: Callable[[str], list[str]]
    run_cloud_debug: Callable[[IncidentRecord], str | None]
    run_smoke: Callable[[IncidentRecord], bool]
    merge_and_pull: Callable[[IncidentRecord], str]
    kill_cell: Callable[[str], None]
    broken_sha: Callable[[], str]
    rescore_all: Callable[[str, str], None]
    sibling_reeval: Callable[[str, str, Sequence[str]], None]
    restart_from_k: Callable[[str, int], None]
    memory_resume: Callable[[str, int], None]


@dataclass
class Orchestrator:
    repo_root: Path
    cell_ids: list[str]
    hooks: BabysitterHooks
    logs: HumanLogWriter = field(init=False)
    store: IncidentStore = field(init=False)
    cells: dict[str, CellRecord] = field(default_factory=dict)
    helper_trace_by_attempt: dict[str, dict[int, list[str]]] = field(default_factory=dict)
    crash_after_cloud_attempts: int | None = None
    max_cloud_attempts: int = MAX_CLOUD_ATTEMPTS
    _cloud_attempts_this_run: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.logs = HumanLogWriter(self.repo_root)
        self.store = IncidentStore(self.repo_root)
        for cell_id in self.cell_ids:
            existing = self.store.load_cell(cell_id)
            self.cells[cell_id] = existing or CellRecord(
                cell_id=cell_id, state=CellState.RUNNING.value
            )
            self.store.save_cell(self.cells[cell_id])

    def set_helper_traces(self, cell_id: str, traces: dict[int, list[str]]) -> None:
        self.helper_trace_by_attempt[cell_id] = dict(traces)

    def _set_state(self, cell_id: str, state: CellState) -> None:
        cell = self.cells[cell_id]
        cell.state = state.value
        self.store.save_cell(cell)

    def _helper_first_use_k(
        self, cell_id: str, attempt_index: int, failing: Sequence[str]
    ) -> int:
        traces = self.helper_trace_by_attempt.get(cell_id) or {}
        failing_set = set(failing)
        candidates = [
            int(idx) for idx, used in traces.items() if failing_set.intersection(used)
        ]
        return min(candidates) if candidates else attempt_index

    def _open_incident(
        self, cell_id: str, attempt_index: int, path_kind: PathKind
    ) -> IncidentRecord:
        cell = self.cells[cell_id]
        if cell.active_incident_id:
            existing = self.store.load_incident(cell.active_incident_id)
            if existing and not existing.closed:
                return existing
        ts = int(time.time())
        incident_id = make_incident_id(cell_id, attempt_index, path_kind, ts)
        record = IncidentRecord(
            incident_id=incident_id,
            cell_id=cell_id,
            attempt_index=attempt_index,
            path_kind=path_kind.value,
            trigger_unix_ts=ts,
            broken_sha=self.hooks.broken_sha(),
            phase=IncidentPhase.OPEN.value,
        )
        cell.active_incident_id = incident_id
        self._set_state(cell_id, CellState.INCIDENT_ACTIVE)
        self.store.save_incident(record)
        self.logs.emit(cell_id, "INCIDENT_OPEN", f"id={incident_id} kind={path_kind.value}")
        return record

    def _kill_once(self, cell_id: str, incident: IncidentRecord) -> None:
        if incident.phase in {
            IncidentPhase.KILLED.value,
            IncidentPhase.CLOUD.value,
            IncidentPhase.SMOKE.value,
            IncidentPhase.MERGE.value,
            IncidentPhase.RECOVERY.value,
        }:
            return
        self.hooks.kill_cell(cell_id)
        self.cells[cell_id].kill_count += 1
        incident.phase = IncidentPhase.KILLED.value
        if not incident.broken_sha:
            incident.broken_sha = self.hooks.broken_sha()
        self.store.save_incident(incident)
        self.store.save_cell(self.cells[cell_id])
        self.logs.emit(cell_id, "KILL", f"incident={incident.incident_id}")

    def _cloud_smoke_merge(self, incident: IncidentRecord) -> bool:
        cell_id = incident.cell_id
        while incident.cloud_attempt_count < self.max_cloud_attempts:
            incident.phase = IncidentPhase.CLOUD.value
            incident.cloud_attempt_count += 1
            self._cloud_attempts_this_run[cell_id] = (
                self._cloud_attempts_this_run.get(cell_id, 0) + 1
            )
            self.store.save_incident(incident)
            self.logs.emit(
                cell_id,
                "CLOUD_DEBUG",
                f"attempt={incident.cloud_attempt_count} kind={incident.path_kind}",
            )
            if (
                self.crash_after_cloud_attempts is not None
                and self._cloud_attempts_this_run.get(cell_id, 0)
                >= self.crash_after_cloud_attempts
            ):
                self.store.save_incident(incident)
                raise RuntimeError("simulated babysitter crash")

            pr_url = self.hooks.run_cloud_debug(incident)
            if pr_url:
                incident.pr_url = pr_url

            incident.phase = IncidentPhase.SMOKE.value
            self.store.save_incident(incident)
            self.logs.emit(cell_id, "SMOKE_START", "model=Qwen2.5-1.5B-Instruct")
            passed = self.hooks.run_smoke(incident)
            if not passed:
                reason = ""
                if incident.extra:
                    reason = str(incident.extra.get("smoke_fail_reason") or "")
                detail = f"attempt={incident.cloud_attempt_count}"
                if reason:
                    detail = f"{detail} reason={reason}"
                self.logs.emit(cell_id, "SMOKE_FAIL", detail)
                # Persist fail reason; do not merge — loop revises via next Cloud attempt.
                self.store.save_incident(incident)
                continue
            self.logs.emit(
                cell_id, "SMOKE_PASS", f"attempt={incident.cloud_attempt_count}"
            )

            incident.phase = IncidentPhase.MERGE.value
            self.store.save_incident(incident)
            sha = self.hooks.merge_and_pull(incident)
            self.logs.emit(cell_id, "MERGED", f"sha={sha}")
            return True

        self._set_state(cell_id, CellState.BLOCKED_NEEDS_HUMAN)
        incident.closed = True
        incident.phase = IncidentPhase.CLOSED.value
        self.store.save_incident(incident)
        self.cells[cell_id].active_incident_id = None
        self.store.save_cell(self.cells[cell_id])
        self.logs.emit(
            cell_id,
            "blocked_needs_human",
            f"cloud_attempts={incident.cloud_attempt_count}",
        )
        return False

    def _close_incident(self, incident: IncidentRecord) -> None:
        incident.closed = True
        incident.phase = IncidentPhase.CLOSED.value
        self.store.save_incident(incident)
        cell = self.cells[incident.cell_id]
        cell.active_incident_id = None
        if cell.state == CellState.INCIDENT_ACTIVE.value:
            cell.state = CellState.RUNNING.value
        self.store.save_cell(cell)
        self.logs.emit(incident.cell_id, "INCIDENT_CLOSE", f"id={incident.incident_id}")

    def _recover_harness_family(self, incident: IncidentRecord) -> None:
        cell_id = incident.cell_id
        sha = incident.broken_sha or "unknown"
        incident.phase = IncidentPhase.RECOVERY.value
        self.store.save_incident(incident)
        self.hooks.rescore_all(cell_id, sha)
        self.logs.emit(cell_id, "RESCORE_ALL", f"broken_sha={sha}")
        siblings = same_domain_siblings(cell_id, self.cell_ids)
        if siblings:
            self.hooks.sibling_reeval(cell_id, sha, siblings)
            self.logs.emit(
                cell_id,
                "SIBLING_REEVAL",
                f"siblings={','.join(siblings)} broken_sha={sha}",
            )

    def _recover_helper(self, incident: IncidentRecord, k: int) -> None:
        cell_id = incident.cell_id
        incident.helper_first_use_k = k
        incident.phase = IncidentPhase.RECOVERY.value
        self.store.save_incident(incident)
        self.cells[cell_id].restart_k = k
        self.store.save_cell(self.cells[cell_id])
        self.logs.emit(cell_id, f"HELPER_FIRST_USE_K={k}")
        self.hooks.restart_from_k(cell_id, k)
        self.logs.emit(cell_id, "RESTART_FROM_K", f"k={k}")

    def _run_fix_path(self, incident: IncidentRecord) -> HandleResult:
        cell_id = incident.cell_id
        path = PathKind(incident.path_kind)
        self._kill_once(cell_id, incident)
        merged = self._cloud_smoke_merge(incident)
        if not merged:
            return HandleResult(woke=True, path_kind=path, blocked=True)

        if path in {PathKind.HARNESS, PathKind.TELEMETRY}:
            self._recover_harness_family(incident)
        elif path in {PathKind.HELPER, PathKind.TIER_B_HELPER}:
            failing = self.hooks.helper_failures(cell_id)
            k = incident.helper_first_use_k or self._helper_first_use_k(
                cell_id, incident.attempt_index, failing
            )
            self._recover_helper(incident, k)
        elif path == PathKind.MEMORY:
            self.hooks.memory_resume(cell_id, incident.attempt_index)
            self.logs.emit(
                cell_id, "MEMORY_OPS_RESUME", f"attempt={incident.attempt_index}"
            )

        self._close_incident(incident)
        self._set_state(cell_id, CellState.RUNNING)
        return HandleResult(woke=True, path_kind=path)

    def handle_memory(self, cell_id: str, attempt_index: int) -> HandleResult:
        # Stale-marker guard at the orchestrator level: memory_resume does not
        # advance the cell log past the old OOM marker, so any caller (not just
        # run_watch_once, whose in-memory dedup can be bypassed by a watcher
        # process running pre-guard code) may re-deliver the same OOM for a
        # (cell, attempt) whose memory incident already closed. Re-opening
        # would kill the healthy resumed run and loop incidents endlessly
        # (incident smiles-acrylates-qwen35-2b:2:memory:1784887455).
        cell = self.cells.get(cell_id)
        has_open = bool(cell and cell.active_incident_id)
        if has_open:
            existing = self.store.load_incident(cell.active_incident_id)
            has_open = bool(existing and not existing.closed)
        if not has_open and self.store.has_closed_incident(
            cell_id, attempt_index, PathKind.MEMORY
        ):
            self.logs.emit(
                cell_id,
                "STALE_MEMORY_EVENT_SKIP",
                f"attempt={attempt_index} scope=orchestrator",
            )
            return HandleResult(woke=False, path_kind=None)
        self.logs.emit(cell_id, "MEMORY_OPS_START", f"attempt={attempt_index}")
        incident = self._open_incident(cell_id, attempt_index, PathKind.MEMORY)
        return self._run_fix_path(incident)

    def handle_accuracy(
        self,
        cell_id: str,
        attempt_index: int,
        accuracy_pct: float | None,
        *,
        finished: bool,
    ) -> HandleResult:
        if not finished:
            return HandleResult(woke=False)

        if accuracy_pct is None:
            self.logs.emit(cell_id, "TELEMETRY_FAIL", f"attempt={attempt_index}")
            incident = self._open_incident(cell_id, attempt_index, PathKind.TELEMETRY)
            return self._run_fix_path(incident)

        if accuracy_pct >= TIER_B_ACC_PCT:
            self.logs.emit(cell_id, "NO_WAKE", f"acc={accuracy_pct}")
            return HandleResult(woke=False)

        if accuracy_pct == 0.0:
            self.logs.emit(cell_id, "TIER_A", f"acc={accuracy_pct}")
            self._set_state(cell_id, CellState.PAUSED_FOR_SUITE)
            twin = self.hooks.twin_accuracy(cell_id)
            failing = self.hooks.helper_failures(cell_id)
            self._set_state(cell_id, CellState.RUNNING)
            if twin <= 0.0:
                self.logs.emit(cell_id, "HARNESS", f"twin_acc={twin}")
                incident = self._open_incident(cell_id, attempt_index, PathKind.HARNESS)
                return self._run_fix_path(incident)
            if failing:
                self.logs.emit(cell_id, "HELPER", f"failures={','.join(failing)}")
                incident = self._open_incident(cell_id, attempt_index, PathKind.HELPER)
                k = self._helper_first_use_k(cell_id, attempt_index, failing)
                incident.helper_first_use_k = k
                self.store.save_incident(incident)
                return self._run_fix_path(incident)
            self.logs.emit(cell_id, "STRATEGY_MISS", f"twin_acc={twin}")
            return HandleResult(woke=False, path_kind=None)

        self.logs.emit(cell_id, "TIER_B", f"acc={accuracy_pct}")
        failing = self.hooks.helper_failures(cell_id)
        if failing:
            self.logs.emit(cell_id, "HELPER", f"failures={','.join(failing)}")
            incident = self._open_incident(
                cell_id, attempt_index, PathKind.TIER_B_HELPER
            )
            k = self._helper_first_use_k(cell_id, attempt_index, failing)
            incident.helper_first_use_k = k
            self.store.save_incident(incident)
            return self._run_fix_path(incident)
        self.logs.emit(cell_id, "LOW_ACC_STRATEGY_MISS", f"acc={accuracy_pct}")
        return HandleResult(woke=False, path_kind=None)

    def handle_event(
        self,
        cell_id: str,
        attempt_index: int,
        *,
        memory_ops: bool = False,
        finished: bool = True,
        accuracy_pct: float | None = None,
        then_accuracy_after_memory: bool = False,
    ) -> HandleResult:
        if memory_ops:
            mem = self.handle_memory(cell_id, attempt_index)
            if mem.blocked:
                return mem
            if then_accuracy_after_memory and finished:
                acc = self.handle_accuracy(
                    cell_id, attempt_index, accuracy_pct, finished=True
                )
                acc.memory_then_accuracy = True
                return acc
            return mem
        return self.handle_accuracy(
            cell_id, attempt_index, accuracy_pct, finished=finished
        )

    def resume_active_incident(self, cell_id: str) -> HandleResult:
        cell = self.cells[cell_id]
        if not cell.active_incident_id:
            return HandleResult(woke=False)
        incident = self.store.load_incident(cell.active_incident_id)
        if incident is None or incident.closed:
            return HandleResult(woke=False)
        self.logs.emit(
            cell_id,
            "INCIDENT_RESUME",
            f"id={incident.incident_id} cloud_attempts={incident.cloud_attempt_count}",
        )
        return self._run_fix_path(incident)
