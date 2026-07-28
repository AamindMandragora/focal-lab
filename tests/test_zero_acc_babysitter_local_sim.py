"""Local simulator coverage for the zero-acc Cloud babysitter (11 scenarios).

These tests drive the state machine only — they never authorize a real queue.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.runtime.zero_acc_babysitter.local_sim import LocalBabysitterSim
from scripts.runtime.zero_acc_babysitter.constants import (
    CellState,
    PathKind,
)


CELL = "gsm-qwen25-1p5b"
SIBLING = "gsm-qwen25-7b"


def _sim(tmp_path: Path, **hooks) -> LocalBabysitterSim:
    return LocalBabysitterSim(
        repo_root=tmp_path,
        cell_ids=[CELL, SIBLING, "spider-qwen25-1p5b"],
        **hooks,
    )


def test_01_tier_a_harness_rescore_all(tmp_path: Path) -> None:
    sim = _sim(tmp_path, twin_accuracy=0.0, helper_failures=[])
    result = sim.finish_attempt(CELL, attempt_index=3, accuracy_pct=0.0)
    assert result.path_kind == PathKind.HARNESS
    assert "TIER_A" in sim.master_text()
    assert "HARNESS" in sim.master_text()
    assert "SMOKE_PASS" in sim.master_text()
    assert "MERGED" in sim.master_text()
    assert "RESCORE_ALL" in sim.master_text()
    assert "SIBLING_REEVAL" in sim.master_text()
    assert sim.cell_state(CELL) == CellState.RUNNING
    assert sim.was_killed(CELL)
    assert sim.merge_count == 1


def test_02_tier_a_helper_restart_from_k(tmp_path: Path) -> None:
    sim = _sim(
        tmp_path,
        twin_accuracy=0.25,
        helper_failures=["HelperFoo"],
        helper_trace_by_attempt={1: [], 2: ["HelperFoo"], 3: ["HelperFoo"]},
    )
    result = sim.finish_attempt(CELL, attempt_index=3, accuracy_pct=0.0)
    assert result.path_kind == PathKind.HELPER
    assert "HELPER_FIRST_USE_K=2" in sim.master_text()
    assert "RESTART_FROM_K" in sim.master_text()
    assert sim.restart_k(CELL) == 2
    assert sim.was_killed(CELL)
    assert sim.merge_count == 1


def test_03_tier_a_strategy_miss_no_kill(tmp_path: Path) -> None:
    sim = _sim(tmp_path, twin_accuracy=0.25, helper_failures=[])
    result = sim.finish_attempt(CELL, attempt_index=2, accuracy_pct=0.0)
    assert result.path_kind is None
    assert "STRATEGY_MISS" in sim.master_text()
    assert not sim.was_killed(CELL)
    assert sim.merge_count == 0
    assert sim.cell_state(CELL) == CellState.RUNNING


def test_04_tier_b_helper_same_cloud_smoke_merge(tmp_path: Path) -> None:
    sim = _sim(
        tmp_path,
        twin_accuracy=0.25,
        helper_failures=["HelperFoo"],
        helper_trace_by_attempt={4: ["HelperFoo"]},
    )
    result = sim.finish_attempt(CELL, attempt_index=4, accuracy_pct=10.0)
    assert result.path_kind == PathKind.TIER_B_HELPER
    assert "TIER_B" in sim.master_text()
    assert "SMOKE_PASS" in sim.master_text()
    assert "MERGED" in sim.master_text()
    assert sim.was_killed(CELL)
    assert sim.merge_count == 1


def test_05_tier_b_low_acc_strategy_miss(tmp_path: Path) -> None:
    sim = _sim(tmp_path, twin_accuracy=0.25, helper_failures=[])
    result = sim.finish_attempt(CELL, attempt_index=4, accuracy_pct=12.5)
    assert result.path_kind is None
    assert "LOW_ACC_STRATEGY_MISS" in sim.master_text()
    assert not sim.was_killed(CELL)
    assert sim.merge_count == 0


def test_06_memory_ops_alone(tmp_path: Path) -> None:
    sim = _sim(tmp_path)
    result = sim.abort_memory(CELL, attempt_index=5)
    assert result.path_kind == PathKind.MEMORY
    assert "MEMORY_OPS_START" in sim.master_text()
    assert "SMOKE_PASS" in sim.master_text()
    assert "MERGED" in sim.master_text()
    assert "MEMORY_OPS_RESUME" in sim.master_text()
    assert sim.was_killed(CELL)
    assert "TIER_A" not in sim.master_text()
    assert "TELEMETRY_FAIL" not in sim.master_text()


def test_07_memory_then_accuracy(tmp_path: Path) -> None:
    sim = _sim(tmp_path, twin_accuracy=0.0, helper_failures=[])
    result = sim.abort_memory_then_finished(
        CELL, attempt_index=6, accuracy_pct=0.0
    )
    assert result.memory_then_accuracy is True
    text = sim.master_text()
    mem_i = text.index("MEMORY_OPS_START")
    tier_i = text.index("TIER_A")
    assert mem_i < tier_i
    assert "HARNESS" in text
    assert sim.merge_count == 2


def test_08_missing_accuracy_telemetry_fail(tmp_path: Path) -> None:
    sim = _sim(tmp_path, twin_accuracy=0.25, helper_failures=[])
    result = sim.finish_attempt(CELL, attempt_index=7, accuracy_pct=None)
    assert result.path_kind == PathKind.TELEMETRY
    assert "TELEMETRY_FAIL" in sim.master_text()
    assert "SMOKE_PASS" in sim.master_text()
    assert "RESCORE_ALL" in sim.master_text()
    assert sim.was_killed(CELL)


def test_09_smoke_fail_retry_then_pass(tmp_path: Path) -> None:
    sim = _sim(
        tmp_path,
        twin_accuracy=0.0,
        helper_failures=[],
        smoke_results=[False, True],
    )
    result = sim.finish_attempt(CELL, attempt_index=1, accuracy_pct=0.0)
    assert result.path_kind == PathKind.HARNESS
    assert "SMOKE_FAIL" in sim.master_text()
    assert "SMOKE_PASS" in sim.master_text()
    assert sim.cloud_attempt_count(CELL) == 2
    assert sim.merge_count == 1
    assert sim.cell_state(CELL) != CellState.BLOCKED_NEEDS_HUMAN


def test_10_acc_ge_15_continue_no_wake(tmp_path: Path) -> None:
    sim = _sim(tmp_path, twin_accuracy=0.0, helper_failures=["HelperFoo"])
    result = sim.finish_attempt(CELL, attempt_index=8, accuracy_pct=15.0)
    assert result.woke is False
    assert "TIER_A" not in sim.master_text()
    assert "TIER_B" not in sim.master_text()
    assert not sim.was_killed(CELL)
    assert sim.cell_state(CELL) == CellState.RUNNING


def test_11_mid_incident_resume_preserves_counter(tmp_path: Path) -> None:
    sim = _sim(
        tmp_path,
        twin_accuracy=0.0,
        helper_failures=[],
        smoke_results=[False, True],
        crash_after_cloud_attempts=1,
    )
    with pytest.raises(RuntimeError, match="simulated babysitter crash"):
        sim.finish_attempt(CELL, attempt_index=1, accuracy_pct=0.0)

    incident_id = sim.active_incident_id(CELL)
    assert incident_id is not None
    assert sim.cloud_attempt_count(CELL) == 1
    assert sim.cell_state(CELL) == CellState.INCIDENT_ACTIVE

    resumed = LocalBabysitterSim.resume_from_disk(
        repo_root=tmp_path,
        cell_ids=[CELL, SIBLING, "spider-qwen25-1p5b"],
        twin_accuracy=0.0,
        helper_failures=[],
        smoke_results=[True],
    )
    assert resumed.active_incident_id(CELL) == incident_id
    assert resumed.cloud_attempt_count(CELL) == 1
    result = resumed.resume_active_incident(CELL)
    assert result.path_kind == PathKind.HARNESS
    assert resumed.cloud_attempt_count(CELL) == 2
    assert resumed.kill_count(CELL) == 1
    assert "SMOKE_PASS" in resumed.master_text()
    assert resumed.cell_state(CELL) == CellState.RUNNING


def test_human_logs_lane_routing(tmp_path: Path) -> None:
    sim = _sim(tmp_path, twin_accuracy=0.0, helper_failures=[])
    sim.finish_attempt(CELL, attempt_index=1, accuracy_pct=0.0)
    assert "TIER_A" in sim.lane_text("gsm")
    assert "TIER_A" not in sim.lane_text("spider")
    assert "TIER_A" not in sim.lane_text("smiles")
    assert (tmp_path / "logs/zero_acc_babysitter/master.log").is_file()


def test_incident_persistence_fields(tmp_path: Path) -> None:
    sim = _sim(tmp_path, twin_accuracy=0.0, helper_failures=[])
    sim.finish_attempt(CELL, attempt_index=9, accuracy_pct=0.0)
    incidents = list((tmp_path / "logs/zero_acc_babysitter/incidents").glob("*.json"))
    assert incidents
    payload = json.loads(incidents[0].read_text(encoding="utf-8"))
    assert payload["path_kind"] == PathKind.HARNESS.value
    assert "broken_sha" in payload
    assert payload["cloud_attempt_count"] >= 1
    cell_side = json.loads(
        (tmp_path / f"logs/zero_acc_babysitter/cells/{CELL}.json").read_text()
    )
    assert cell_side["state"] == CellState.RUNNING.value


def test_max_30_cloud_attempts_blocks(tmp_path: Path) -> None:
    sim = _sim(
        tmp_path,
        twin_accuracy=0.0,
        helper_failures=[],
        smoke_results=[False] * 30,
    )
    result = sim.finish_attempt(CELL, attempt_index=1, accuracy_pct=0.0)
    assert sim.cell_state(CELL) == CellState.BLOCKED_NEEDS_HUMAN
    assert sim.cloud_attempt_count(CELL) == 30
    assert result.blocked is True
