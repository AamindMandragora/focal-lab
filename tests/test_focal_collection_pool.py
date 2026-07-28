from __future__ import annotations

import importlib.util
from collections import deque
from pathlib import Path


def load_pool_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_focal_collection_pool.py"
    spec = importlib.util.spec_from_file_location("run_focal_collection_pool", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manifest_covers_remaining_collection_exactly_once(tmp_path):
    pool = load_pool_module()

    jobs = pool.build_manifest(tmp_path)
    labels = [job.label for job in jobs]

    expected_4b = {f"cars-4b-{start:03d}-{start + 25:03d}" for start in range(125, 300, 25)}
    expected_9b = {f"cars-9b-{start:03d}-{start + 10:03d}" for start in range(0, 300, 10)}
    assert expected_4b <= set(labels)
    assert expected_9b <= set(labels)
    assert len(labels) == len(set(labels)) == 44


def test_manifest_uses_cars_huggingface_and_compatible_vllm_jobs(tmp_path):
    pool = load_pool_module()
    jobs = pool.build_manifest(tmp_path)

    cars_jobs = [job for job in jobs if job.label.startswith("cars-")]
    vllm_jobs = [job for job in jobs if not job.label.startswith("cars-")]

    assert len(cars_jobs) == 37
    assert all("--eval-backend" in job.args and "huggingface" in job.args for job in cars_jobs)
    assert all("--device" in job.args and "cuda" in job.args for job in cars_jobs)
    assert {job.estimated_memory_mib for job in cars_jobs if job.label.startswith("cars-4b-")} == {10_000}
    assert {job.estimated_memory_mib for job in cars_jobs if job.label.startswith("cars-9b-")} == {19_000}
    assert all(not job.exclusive_gpu for job in cars_jobs)
    assert len(vllm_jobs) == 7
    assert all("--eval-backend" in job.args and "vllm" in job.args for job in vllm_jobs)
    crane_job = next(job for job in vllm_jobs if job.label == "spider-9b-crane")
    assert crane_job.estimated_memory_mib == 19_000
    assert crane_job.exclusive_gpu is False
    exclusive_vllm_jobs = [job for job in vllm_jobs if job.label != "spider-9b-crane"]
    assert all(job.estimated_memory_mib == 39_000 for job in exclusive_vllm_jobs)
    assert all(job.exclusive_gpu for job in exclusive_vllm_jobs)
    assert all("AWS_BEARER_TOKEN_BEDROCK" not in " ".join(job.args) for job in jobs)


def test_ready_gpus_are_greedy_and_exclude_busy_or_used_devices():
    pool = load_pool_module()

    ready = pool.ready_gpu_ids(
        memory_used_mib={0: 10, 1: 8000, 2: 4, 3: 10},
        busy_gpu_ids={2},
        max_idle_memory_mib=1000,
    )

    assert ready == [0, 3]


def test_claim_fitting_job_skips_finished_and_bypasses_oversized_head(tmp_path):
    pool = load_pool_module()
    done = tmp_path / "done.json"
    done.write_text("{}", encoding="utf-8")
    large_output = tmp_path / "large.json"
    small_output = tmp_path / "small.json"
    queue = deque(
        [
            pool.Job("done", done, tmp_path / "done.log", ("true",)),
            pool.Job("large", large_output, tmp_path / "large.log", ("true",), estimated_memory_mib=19_000),
            pool.Job("small", small_output, tmp_path / "small.log", ("true",), estimated_memory_mib=10_000),
        ]
    )

    job, skipped = pool.claim_fitting_job(
        queue,
        used_memory_mib=28_000,
        total_memory_mib=40_960,
        safety_memory_mib=2_000,
        idle_memory_mib=1_000,
    )

    assert job.label == "small"
    assert skipped == ["done"]
    assert [queued.label for queued in queue] == ["large"]
    assert job.output_json.with_name(f"{job.output_json.name}.running").is_dir()


def test_claim_fitting_job_leaves_manually_claimed_output_queued(tmp_path):
    pool = load_pool_module()
    output = tmp_path / "crane.json"
    output.with_name(f"{output.name}.running").mkdir()
    queue = deque(
        [
            pool.Job(
                "crane",
                output,
                tmp_path / "crane.log",
                ("true",),
                estimated_memory_mib=19_000,
            )
        ]
    )

    job, skipped = pool.claim_fitting_job(
        queue,
        used_memory_mib=17_800,
        total_memory_mib=40_960,
        safety_memory_mib=2_000,
        idle_memory_mib=1_000,
    )

    assert job is None
    assert skipped == []
    assert [queued.label for queued in queue] == ["crane"]


def test_exclusive_job_requires_an_idle_gpu(tmp_path):
    pool = load_pool_module()
    queue = deque(
        [
            pool.Job(
                "vllm",
                tmp_path / "vllm.json",
                tmp_path / "vllm.log",
                ("true",),
                estimated_memory_mib=39_000,
                exclusive_gpu=True,
            )
        ]
    )

    job, _ = pool.claim_fitting_job(
        queue,
        used_memory_mib=8_000,
        total_memory_mib=40_960,
        safety_memory_mib=2_000,
        idle_memory_mib=1_000,
    )

    assert job is None


def test_external_reservations_release_and_requeue_missing_output(tmp_path):
    pool = load_pool_module()
    failed_output = tmp_path / "failed.json"
    done_output = tmp_path / "done.json"
    done_output.write_text("{}", encoding="utf-8")
    failed_job = pool.Job("failed", failed_output, tmp_path / "failed.log", ("false",))
    done_job = pool.Job("done", done_output, tmp_path / "done.log", ("true",))
    external = {
        101: pool.ExternalJob(101, 0, 10_000, failed_output),
        102: pool.ExternalJob(102, 1, 19_000, done_output),
    }
    queue = deque()

    reservations = pool.external_reserved_memory(external, [0, 1, 2, 3])
    pool.reconcile_external_jobs(
        external,
        queue,
        {failed_output.resolve(): failed_job, done_output.resolve(): done_job},
        is_alive=lambda _pid: False,
    )

    assert reservations == {0: 10_000, 1: 19_000, 2: 0, 3: 0}
    assert not external
    assert [job.label for job in queue] == ["failed"]
    assert queue[0].attempt == 1


def test_failed_job_requeues_once_then_stays_failed(tmp_path):
    pool = load_pool_module()
    queue = deque()
    job = pool.Job("oom", tmp_path / "out.json", tmp_path / "out.log", ("false",))

    assert pool.requeue_failed_job(job, queue, max_retries=1) is True
    retried = queue.popleft()
    assert retried.attempt == 1
    assert pool.requeue_failed_job(retried, queue, max_retries=1) is False
    assert not queue


def test_release_job_claim_removes_scheduler_claim(tmp_path):
    pool = load_pool_module()
    job = pool.Job("crane", tmp_path / "crane.json", tmp_path / "crane.log", ("true",))
    claim = pool.job_claim_path(job)
    claim.mkdir()

    pool.release_job_claim(job)

    assert not claim.exists()


def test_release_job_claim_retains_claim_while_process_group_lives(tmp_path, monkeypatch):
    pool = load_pool_module()
    job = pool.Job("crane", tmp_path / "crane.json", tmp_path / "crane.log", ("true",))
    claim = pool.job_claim_path(job)
    claim.mkdir()
    monkeypatch.setattr(pool, "process_group_is_alive", lambda _process_group_id: True)

    released = pool.release_job_claim(job, process_group_id=123)

    assert released is False
    assert claim.is_dir()
