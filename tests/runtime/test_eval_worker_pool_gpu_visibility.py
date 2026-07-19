from types import SimpleNamespace

from synthesis.scripts import eval_worker_pool
from synthesis.scripts import sharded_eval_core


def _four_idle_gpus(*_args, **_kwargs):
    return SimpleNamespace(
        stdout=(
            "0, 10, 40960, 0\n"
            "1, 10, 40960, 0\n"
            "2, 10, 40960, 0\n"
            "3, 10, 40960, 0\n"
        )
    )


def test_gpu_slot_detection_respects_cuda_visible_devices(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.setattr(sharded_eval_core.subprocess, "run", _four_idle_gpus)

    slots = sharded_eval_core.detect_gpu_slots(
        workers_per_gpu=1,
        idle_util_threshold=30,
        min_free_mb=8000,
    )

    assert slots == [3]


def test_worker_pool_fallback_stays_on_the_assigned_visible_gpu(monkeypatch):
    created_gpus = []

    class FakeWorker:
        def __init__(self, worker_id, gpu):
            created_gpus.append((worker_id, gpu))

        def configure(self, _config):
            pass

        def shutdown(self):
            pass

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.delenv("CSD_EVAL_POOL_SIZE", raising=False)
    monkeypatch.setattr(eval_worker_pool, "detect_gpu_slots", lambda *_args: [])
    monkeypatch.setattr(eval_worker_pool, "_Worker", FakeWorker)

    pool = eval_worker_pool.EvalWorkerPool({})

    assert created_gpus == [(0, 2)]
    pool.shutdown()
