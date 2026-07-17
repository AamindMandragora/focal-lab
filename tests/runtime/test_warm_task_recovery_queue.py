import json
from pathlib import Path

import pytest

from scripts.runtime.run_warm_task_recovery_queue import (
    ConfigError,
    choose_gpu,
    load_manifest,
    run_job,
    worker_environment,
)


def _job(**overrides):
    job = {
        "cell_id": "gsm-qwen35-2b",
        "last_clean_attempt": 10,
        "total_cap": 40,
        "memory_reservation_mib": 13000,
        "source_log": "/repo/outputs/source/run.log",
        "history_file": "/repo/.context/history.json",
        "output_name": "warmfix_gsm-qwen35-2b_0714",
        "log_file": "/repo/logs/paid_synth_warmfix_gsm-qwen35-2b.log",
        "dataset": "gsm_symbolic",
        "eval_model": "Qwen/Qwen3.5-2B",
        "gpu_mem_util": 0.40,
        "heldout_sample_size": 49,
        "eval_max_steps": 900,
        "eval_max_seconds": 600,
        "heldout_split_name": "eval",
        "heldout_output_json": "/repo/outputs/reeval/fake.json",
    }
    job.update(overrides)
    return job


def test_manifest_rejects_attempt_after_total_cap(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps([_job(last_clean_attempt=41)]))

    with pytest.raises(ConfigError, match="last_clean_attempt"):
        load_manifest(path)


def test_worker_environment_replays_last_clean_attempt_on_assigned_gpu():
    env = worker_environment(
        _job(seed_file="/repo/.context/attempt10.dfy"),
        assigned_gpu=2,
        inherited={"PATH": "/bin"},
    )

    assert env["RESUME_LAST_ATTEMPT"] == "10"
    assert env["RESUME_TOTAL_CAP"] == "40"
    assert env["RESUME_GPU"] == "2"
    assert env["RESUME_OUTPUT_NAME"] == "warmfix_gsm-qwen35-2b_0714"
    assert env["RESUME_SOURCE_LOG"].endswith("/outputs/source/run.log")
    assert env["RESUME_HISTORY_FILE"].endswith("/.context/history.json")
    assert env["RESUME_SEED_FILE"].endswith("/.context/attempt10.dfy")


def test_choose_gpu_accounts_for_existing_reservations_and_safety_margin():
    snapshots = {
        0: {"used_mib": 10, "total_mib": 40960},
        1: {"used_mib": 10, "total_mib": 40960},
    }
    reservations = {0: {"gsm9": 25000}, 1: {"gsm14": 35000}}
    baseline = {gpu: dict(snapshot) for gpu, snapshot in snapshots.items()}

    # The 2B worker asks vLLM for 40% of the physical GPU (16,384 MiB),
    # so its stale 13,000 MiB manifest estimate must not let it share GPU 0
    # with a 25,000 MiB worker.
    assert choose_gpu(_job(), snapshots, reservations, baseline) is None
    assert choose_gpu(
        _job(memory_reservation_mib=19000), snapshots, reservations, baseline
    ) is None


def test_choose_gpu_uses_vllm_fraction_when_it_exceeds_manifest_estimate():
    snapshots = {
        0: {"used_mib": 10, "total_mib": 40960},
        1: {"used_mib": 10, "total_mib": 40960},
    }
    reservations = {0: {"gsm9": 25000}, 1: {}}
    baseline = {gpu: dict(snapshot) for gpu, snapshot in snapshots.items()}

    assert choose_gpu(_job(), snapshots, reservations, baseline) == 1


def test_manifest_requires_unique_cells_and_existing_checkpoint_inputs(tmp_path):
    source = tmp_path / "run.log"
    history = tmp_path / "history.json"
    source.write_text("source")
    history.write_text("[]")
    duplicate = _job(source_log=str(source), history_file=str(history))
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps([duplicate, duplicate]))

    with pytest.raises(ConfigError, match="duplicate cell_id"):
        load_manifest(path)


def test_run_job_resumes_heldout_from_persisted_success_without_synthesis(
    tmp_path, monkeypatch
):
    job = _job(
        output_name="warmfix-resume-heldout",
        log_file=str(tmp_path / "logs" / "worker.log"),
        heldout_output_json=str(tmp_path / "heldout.json"),
    )
    run_dir = tmp_path / "outputs" / "generated" / job["output_name"] / "run-1"
    compiled = tmp_path / "compiled"
    (run_dir / "results").mkdir(parents=True)
    compiled.mkdir()
    (compiled / "GeneratedCSD.py").write_text("# compiled")
    (run_dir / "results" / "success_report.json").write_text(
        json.dumps({"compiled_dir": str(compiled)})
    )
    latest = run_dir.parent / "latest_run.txt"
    latest.write_text(str(run_dir))
    (tmp_path / "logs").mkdir()
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr("scripts.runtime.run_warm_task_recovery_queue.subprocess.run", fake_run)

    assert run_job(
        job,
        2,
        repo=tmp_path,
        resume_script=tmp_path / "resume.sh",
        python=tmp_path / "python",
        dry_run=False,
    ) == 0
    assert len(commands) == 1
    assert commands[0][0] == str(tmp_path / "python")
    assert "synthesis.scripts.reevaluate_compiled_csd" in commands[0]
