from pathlib import Path

import pytest

from scripts.runtime import build_full_baseline_cold_manifest as builder
from scripts.runtime import run_cold_synthesis_queue as queue


def _job(cell_id: str) -> dict:
    return {
        "cell_id": cell_id,
        "dataset": "smiles",
        "memory_reservation_mib": 40_000,
        "gpu_mem_util": 0.8,
    }


def test_active_exact_zero_marker_blocks_queue_startup(tmp_path: Path) -> None:
    marker = tmp_path / ".context" / "exact-zero-repair-synthesis.blocked"
    marker.parent.mkdir(parents=True)
    marker.write_text("blocked", encoding="utf-8")

    with pytest.raises(queue.ConfigError, match="exact-zero baseline repair"):
        queue.require_synthesis_unblocked(tmp_path)
@pytest.mark.parametrize("marker_kind", ["directory", "dangling-symlink"])
def test_any_existing_exact_zero_marker_blocks_queue_startup(
    tmp_path: Path, marker_kind: str
) -> None:
    marker = tmp_path / ".context" / "exact-zero-repair-synthesis.blocked"
    marker.parent.mkdir(parents=True)
    if marker_kind == "directory":
        marker.mkdir()
    else:
        marker.symlink_to("missing-target")

    with pytest.raises(queue.ConfigError, match="exact-zero baseline repair"):
        queue.require_synthesis_unblocked(tmp_path)




def test_builder_uses_the_same_synthesis_block_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checked: list[Path] = []
    monkeypatch.setattr(
        queue,
        "require_synthesis_unblocked",
        lambda repo: checked.append(Path(repo)),
    )

    builder.require_synthesis_unblocked(tmp_path)

    assert checked == [tmp_path]


def test_dispatch_runs_the_guard_before_every_worker_launch() -> None:
    guarded: list[str] = []
    started: list[str] = []

    queue.dispatch(
        [_job("first"), _job("second")],
        snapshot=lambda: {0: {"used_mib": 0, "total_mib": 48_000}},
        worker=lambda job, _gpus: started.append(job["cell_id"]) or 0,
        poll_seconds=0.001,
        launch_guard=lambda: guarded.append("checked"),
    )

    assert guarded == ["checked", "checked"]
    assert started == ["first", "second"]


def test_dispatch_guard_failure_prevents_worker_launch() -> None:
    started: list[str] = []

    def fail_closed() -> None:
        raise queue.ConfigError("blocked before launch")

    with pytest.raises(queue.ConfigError, match="blocked before launch"):
        queue.dispatch(
            [_job("first")],
            snapshot=lambda: {0: {"used_mib": 0, "total_mib": 48_000}},
            worker=lambda job, _gpus: started.append(job["cell_id"]) or 0,
            poll_seconds=0.001,
            launch_guard=fail_closed,
        )

    assert started == []
