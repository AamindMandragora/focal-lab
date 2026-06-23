from __future__ import annotations

import os

import pytest

from synthesis.storage_env import ensure_repo_cache_env, ensure_repo_outputs_env, ensure_shared_storage_env


@pytest.fixture(autouse=True)
def _isolate_storage_env(monkeypatch):
    keys = (
        "CSD_CACHE_ROOT",
        "CSD_OUTPUTS_ROOT",
        "CSD_OUTPUT_DIR",
        "CSD_BASELINE_OUTPUT_DIR",
        "CSD_ABLATION_OUTPUT_DIR",
        "CSD_GPU3_RETRY_QUEUE",
        "HF_HOME",
        "HF_CACHE",
        "TRANSFORMERS_CACHE",
        "SYNCODE_CACHE",
        "ITER_SYNCODE_CACHE",
        "CSD_LOGS_DIR",
    )
    saved = {key: os.environ.pop(key, None) for key in keys}
    yield
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def test_outputs_root_propagates_subdirs(tmp_path, monkeypatch):
    shared = tmp_path / "team_outputs"
    monkeypatch.setenv("CSD_OUTPUTS_ROOT", str(shared))

    root = ensure_repo_outputs_env()

    assert root == shared.resolve()
    assert os.environ["CSD_OUTPUT_DIR"] == str(shared / "generated")
    assert os.environ["CSD_BASELINE_OUTPUT_DIR"] == str(shared / "baselines")
    assert os.environ["CSD_ABLATION_OUTPUT_DIR"] == str(shared / "ablations")
    assert os.environ["CSD_GPU3_RETRY_QUEUE"] == str(shared / "gpu3_retry_queue.jsonl")
    assert (shared / "generated").is_dir()
    assert "CSD_LOGS_DIR" not in os.environ


def test_outputs_root_does_not_override_explicit_subdir(tmp_path, monkeypatch):
    shared = tmp_path / "team_outputs"
    custom = tmp_path / "custom_baselines"
    monkeypatch.setenv("CSD_OUTPUTS_ROOT", str(shared))
    monkeypatch.setenv("CSD_BASELINE_OUTPUT_DIR", str(custom))

    ensure_repo_outputs_env()

    assert os.environ["CSD_BASELINE_OUTPUT_DIR"] == str(custom)


def test_cache_root_propagates_hf_and_syncode(tmp_path, monkeypatch):
    shared = tmp_path / "team_cache"
    monkeypatch.setenv("CSD_CACHE_ROOT", str(shared))

    root = ensure_repo_cache_env()

    assert root == shared.resolve()
    assert os.environ["HF_HOME"] == str(shared)
    assert os.environ["SYNCODE_CACHE"] == str(shared) + os.sep


def test_shared_storage_env_returns_both_roots(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    outputs = tmp_path / "outputs"
    monkeypatch.setenv("CSD_CACHE_ROOT", str(cache))
    monkeypatch.setenv("CSD_OUTPUTS_ROOT", str(outputs))

    got_cache, got_outputs = ensure_shared_storage_env()

    assert got_cache == cache.resolve()
    assert got_outputs == outputs.resolve()
