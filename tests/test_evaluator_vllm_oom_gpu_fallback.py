"""vLLM evaluator startup OOM fallback: move to the freest GPU when tp==1.

Regression test for incident spider-qwen25-1p5b:3:memory:1784872216 —
with multiple visible GPUs and tensor_parallel_size=1, vLLM always started
on visible device 0 even when a sibling process occupied it, and the retry
loop only shrank gpu_memory_utilization on that same busy device until the
sampler warm-up OOMed. The retry loop must instead narrow
CUDA_VISIBLE_DEVICES to the freest GPU and restart the utilization ladder.
"""

import sys
import types

import pytest

import synthesis.evaluate.benchmarks.common.model_utils as model_utils
import synthesis.evaluate.benchmarks.sql_spider.environment as spider_env
from synthesis.evaluate.evaluator import Evaluator


STARTUP_OOM = ValueError(
    "Free memory on device cuda:0 (17.81/39.49 GiB) on startup is less than "
    "desired GPU memory utilization (0.55, 21.72 GiB)."
)


@pytest.fixture()
def evaluator():
    return Evaluator(
        dataset_name="spider",
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        backend="vllm",
        device="cuda",
        vllm_tensor_parallel_size=1,
        vllm_gpu_memory_utilization=0.6,
    )


def _patch_common(monkeypatch, narrowed_calls):
    monkeypatch.setattr(model_utils, "clear_vllm_engine_cache", lambda: None)
    monkeypatch.setattr(
        model_utils, "pick_cuda_device_index_with_most_free_memory", lambda: 1
    )

    def fake_narrow(idx):
        narrowed_calls.append(idx)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
        return "2"

    monkeypatch.setattr(
        model_utils, "narrow_cuda_visible_devices_to_index", fake_narrow
    )
    torch_stub = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False)
    )
    monkeypatch.setitem(sys.modules, "torch", torch_stub)


def test_tp1_memory_error_narrows_to_freest_gpu(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,2,1")
    narrowed_calls = []
    _patch_common(monkeypatch, narrowed_calls)

    attempts = []

    def fake_setup(**kwargs):
        attempts.append(
            (
                kwargs["vllm_gpu_memory_utilization"],
                model_utils.visible_cuda_device_ids(),
            )
        )
        if len(model_utils.visible_cuda_device_ids()) > 1:
            raise STARTUP_OOM
        return {"env": "ok"}

    monkeypatch.setattr(spider_env, "setup_dafny_environment", fake_setup)

    ev = Evaluator(
        dataset_name="spider",
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        backend="vllm",
        device="cuda",
        vllm_tensor_parallel_size=1,
        vllm_gpu_memory_utilization=0.6,
    )
    env = ev._setup_environment(tmp_path / "generated_csd" / "module.py")

    assert env == {"env": "ok"}
    assert narrowed_calls == [1]
    # First attempt on the shared visible set fails, then the ladder restarts
    # from the requested utilization on the narrowed (free) device.
    assert attempts[0] == (0.6, ["3", "2", "1"])
    assert attempts[1] == (0.6, ["2"])
    assert len(attempts) == 2


def test_single_visible_gpu_still_walks_util_ladder(monkeypatch, tmp_path, evaluator):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    narrowed_calls = []
    _patch_common(monkeypatch, narrowed_calls)

    utils_tried = []

    def fake_setup(**kwargs):
        util = kwargs["vllm_gpu_memory_utilization"]
        utils_tried.append(util)
        if util > 0.5:
            raise STARTUP_OOM
        return {"env": "ok"}

    monkeypatch.setattr(spider_env, "setup_dafny_environment", fake_setup)

    env = evaluator._setup_environment(tmp_path / "generated_csd" / "module.py")

    assert env == {"env": "ok"}
    assert narrowed_calls == []
    assert utils_tried == [0.6, 0.55, 0.5]


def test_non_memory_error_is_raised(monkeypatch, tmp_path, evaluator):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,2,1")
    narrowed_calls = []
    _patch_common(monkeypatch, narrowed_calls)

    def fake_setup(**kwargs):
        raise RuntimeError("unrelated failure")

    monkeypatch.setattr(spider_env, "setup_dafny_environment", fake_setup)

    with pytest.raises(RuntimeError, match="unrelated failure"):
        evaluator._setup_environment(tmp_path / "generated_csd" / "module.py")
    assert narrowed_calls == []
