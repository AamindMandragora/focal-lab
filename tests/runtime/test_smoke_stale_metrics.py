"""Stale smoke metrics from a prior cloud attempt must trigger a fresh smoke run.

Regression for incident spider-qwen25-1p5b:2:telemetry:1784857356: attempt 2's
smoke failed (rc=2) and attached metrics to incident.extra; attempts 3-4 then
reused those stale metrics and re-failed instantly without ever re-running the
smoke on the new PR tip.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.runtime.zero_acc_babysitter import production_hooks
from scripts.runtime.zero_acc_babysitter.persistence import IncidentRecord
from scripts.runtime.zero_acc_babysitter.smoke import SmokeDecision, SmokeMetrics


def _incident(**overrides) -> IncidentRecord:
    kwargs = dict(
        incident_id="spider-x:2:telemetry:123",
        cell_id="spider-qwen25-1p5b",
        attempt_index=2,
        path_kind="telemetry",
        trigger_unix_ts=123,
        cloud_attempt_count=3,
        extra={},
    )
    kwargs.update(overrides)
    return IncidentRecord(**kwargs)


def test_stale_metrics_from_prior_attempt_rerun_smoke(monkeypatch, tmp_path):
    incident = _incident(
        extra={
            "smoke_report_path": str(tmp_path / "missing" / "smoke_report.json"),
            "smoke_process_rc": 2,
            "smoke_accuracy_line_present": False,
            "smoke_attempt": 2,
        }
    )
    reran = []

    def fake_job(inc, **_kwargs):
        reran.append(inc.cloud_attempt_count)
        return SmokeDecision(True, "fresh smoke ran", SmokeMetrics(process_rc=0))

    monkeypatch.setattr(production_hooks, "run_hardened_smoke_job", fake_job)
    decide = production_hooks.make_smoke_decide(
        live_repo=tmp_path, repair_worktree=tmp_path
    )
    decision = decide(incident)
    assert reran == [3]
    assert decision.passed is True


def test_current_attempt_metrics_are_used_without_rerun(monkeypatch, tmp_path):
    incident = _incident(
        extra={
            "smoke_report_path": str(tmp_path / "missing" / "smoke_report.json"),
            "smoke_process_rc": 2,
            "smoke_accuracy_line_present": False,
            "smoke_attempt": 3,
        }
    )

    def fail_if_called(*_a, **_k):  # pragma: no cover
        pytest.fail("smoke job must not re-run for current-attempt metrics")

    monkeypatch.setattr(production_hooks, "run_hardened_smoke_job", fail_if_called)
    decide = production_hooks.make_smoke_decide(
        live_repo=tmp_path, repair_worktree=tmp_path
    )
    decision = decide(incident)
    assert decision.passed is False
    assert "process_rc=2" in decision.reason


def test_attach_smoke_report_stamps_attempt(tmp_path):
    incident = _incident()
    report = tmp_path / "smoke_report.json"
    report.write_text('{"accuracy": 0.5}', encoding="utf-8")
    production_hooks.attach_smoke_report(incident, report, process_rc=0)
    assert incident.extra["smoke_attempt"] == 3
