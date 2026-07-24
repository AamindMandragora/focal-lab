"""Tier-A twin must be a real measurement, not the hardcoded 0.0 stub.

Regression for incident spider-qwen25-1p5b:3:harness:1784874093: production
wired default_stub_twin_accuracy (always 0.0), so every Tier A wake was
classified as a harness failure even when the harness measurably worked.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.runtime.zero_acc_babysitter.smoke as smoke
from scripts.runtime.zero_acc_babysitter.production_hooks import (
    build_production_hooks,
    make_twin_accuracy,
)
from scripts.runtime.zero_acc_babysitter.smoke import run_twin_accuracy_probe

CELL = "spider-qwen25-1p5b"


@pytest.fixture(autouse=True)
def _pin_smoke_gpu(monkeypatch):
    # Keep tests off the real nvidia-smi wait loop (can sleep on a busy host).
    monkeypatch.setattr(smoke, "wait_for_smoke_gpu", lambda **_kw: "2")


def _seed_csd(live: Path) -> Path:
    csd = live / "outputs" / "generated" / f"coldq_{CELL}_x" / "GeneratedCSD.py"
    csd.parent.mkdir(parents=True)
    csd.write_text("# compiled csd\n", encoding="utf-8")
    return csd


def _runner_writing(accuracy: float | None, rc: int = 0):
    def _runner(cmd, *, cwd, env, out_dir):
        if accuracy is not None:
            report = Path(cmd[cmd.index("--output-json") + 1])
            report.write_text(json.dumps({"accuracy": accuracy}), encoding="utf-8")
        return rc

    return _runner


def test_probe_returns_measured_accuracy_pct(tmp_path):
    _seed_csd(tmp_path)
    acc = run_twin_accuracy_probe(
        CELL, live_repo=tmp_path, runner=_runner_writing(0.8)
    )
    assert acc == 80.0


def test_probe_none_without_generated_csd(tmp_path):
    assert (
        run_twin_accuracy_probe(CELL, live_repo=tmp_path, runner=_runner_writing(0.8))
        is None
    )


def test_probe_none_on_nonzero_rc(tmp_path):
    _seed_csd(tmp_path)
    acc = run_twin_accuracy_probe(
        CELL, live_repo=tmp_path, runner=_runner_writing(0.8, rc=1)
    )
    assert acc is None


def test_make_twin_accuracy_maps_failure_to_zero(tmp_path):
    twin = make_twin_accuracy(live_repo=tmp_path, runner=_runner_writing(None, rc=1))
    assert twin(CELL) == 0.0


def test_production_hooks_default_twin_measures(tmp_path):
    _seed_csd(tmp_path)

    class _NullClient:
        def debug_fix(self, incident):
            return None

    hooks = build_production_hooks(
        live_repo=tmp_path,
        repair_worktree=tmp_path / "repair",
        repair_client=_NullClient(),
        cell_ids=[CELL],
        smoke_job_runner=_runner_writing(0.8),
    )
    assert hooks.twin_accuracy(CELL) == 80.0
