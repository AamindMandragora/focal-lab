"""Smoke GPU jobs must survive transient GPU contention.

Regression for incident spider-qwen25-1p5b:3:harness:1784874093: the smoke
scanned GPU 2 at 19 GiB free, another queue job loaded its vLLM engine in the
scan-to-init window, and every engine retry died with "Free memory on device
cuda:0 (4.5/39.49 GiB) ... desired 11.85 GiB" -> smoke rc=1 -> a cloud repair
attempt burned on pure contention. The smoke must (a) only pick a GPU that
currently fits its engine, waiting for one if needed, and (b) retry locally
when the failure log shows an engine-init contention signature.
"""

from __future__ import annotations

import json
from pathlib import Path

import scripts.runtime.zero_acc_babysitter.smoke as smoke
from scripts.runtime.zero_acc_babysitter.smoke import (
    run_twin_accuracy_probe,
    smoke_gpu_required_free_mb,
    wait_for_smoke_gpu,
)

CELL = "spider-qwen25-1p5b"

CONTENTION_LOG = (
    "ValueError: Free memory on device cuda:0 (4.5/39.49 GiB) on startup is "
    "less than desired GPU memory utilization (0.3, 11.85 GiB).\n"
    "Evaluation failed: Engine core initialization failed. See root cause "
    "above. Failed core proc(s): {}\n"
)


def _seed_csd(live: Path) -> None:
    csd = live / "outputs" / "generated" / f"coldq_{CELL}_x" / "GeneratedCSD.py"
    csd.parent.mkdir(parents=True)
    csd.write_text("# compiled csd\n", encoding="utf-8")


def test_wait_for_smoke_gpu_skips_gpu_without_room_for_engine(monkeypatch):
    # GPU 2 shows plenty free by the old ">= 8 GB idle" notion but not enough
    # for the smoke engine; GPU 1 fits.
    need = smoke_gpu_required_free_mb(40437)
    rows = [("1", need + 500, 40437), ("2", need - 500, 40437)]
    monkeypatch.setattr(smoke, "_allowed_gpu_inventory", lambda: rows)
    assert wait_for_smoke_gpu() == "1"


def test_wait_for_smoke_gpu_polls_until_a_gpu_frees_up(monkeypatch):
    need = smoke_gpu_required_free_mb(40437)
    states = iter(
        [
            [("2", 4500, 40437)],
            [("2", 4500, 40437)],
            [("2", need + 1000, 40437)],
        ]
    )
    monkeypatch.setattr(smoke, "_allowed_gpu_inventory", lambda: next(states))
    sleeps: list[float] = []
    gpu = wait_for_smoke_gpu(
        max_wait_seconds=900.0,
        poll_seconds=30.0,
        sleep=sleeps.append,
        now=lambda: 0.0,
    )
    assert gpu == "2"
    assert sleeps == [30.0, 30.0]


def test_wait_for_smoke_gpu_falls_back_to_freest_at_deadline(monkeypatch):
    rows = [("1", 3000, 40437), ("2", 4500, 40437)]
    monkeypatch.setattr(smoke, "_allowed_gpu_inventory", lambda: rows)
    clock = iter([0.0, 0.0, 1000.0])
    gpu = wait_for_smoke_gpu(
        max_wait_seconds=900.0,
        poll_seconds=30.0,
        sleep=lambda _s: None,
        now=lambda: next(clock),
    )
    assert gpu == "2"


def test_twin_probe_retries_after_gpu_contention_then_measures(tmp_path, monkeypatch):
    _seed_csd(tmp_path)
    monkeypatch.setattr(smoke, "wait_for_smoke_gpu", lambda **_kw: "2")
    calls: list[str] = []

    def _runner(cmd, *, cwd, env, out_dir):
        calls.append(env["CUDA_VISIBLE_DEVICES"])
        if len(calls) == 1:
            (out_dir / "smoke.log").write_text(CONTENTION_LOG, encoding="utf-8")
            return 1
        report = Path(cmd[cmd.index("--output-json") + 1])
        report.write_text(json.dumps({"accuracy": 0.8}), encoding="utf-8")
        return 0

    acc = run_twin_accuracy_probe(CELL, live_repo=tmp_path, runner=_runner)
    assert acc == 80.0
    assert calls == ["2", "2"]


def test_twin_probe_does_not_retry_non_contention_failures(tmp_path, monkeypatch):
    _seed_csd(tmp_path)
    monkeypatch.setattr(smoke, "wait_for_smoke_gpu", lambda **_kw: "2")
    calls: list[int] = []

    def _runner(cmd, *, cwd, env, out_dir):
        calls.append(1)
        (out_dir / "smoke.log").write_text("Traceback: bad CSD\n", encoding="utf-8")
        return 1

    assert run_twin_accuracy_probe(CELL, live_repo=tmp_path, runner=_runner) is None
    assert len(calls) == 1


def test_twin_probe_pins_cuda_visible_devices_over_inherited_env(
    tmp_path, monkeypatch
):
    _seed_csd(tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,2")
    monkeypatch.setattr(smoke, "wait_for_smoke_gpu", lambda **_kw: "2")
    seen: list[str] = []

    def _runner(cmd, *, cwd, env, out_dir):
        seen.append(env["CUDA_VISIBLE_DEVICES"])
        report = Path(cmd[cmd.index("--output-json") + 1])
        report.write_text(json.dumps({"accuracy": 0.4}), encoding="utf-8")
        return 0

    assert run_twin_accuracy_probe(CELL, live_repo=tmp_path, runner=_runner) == 40.0
    assert seen == ["2"]
