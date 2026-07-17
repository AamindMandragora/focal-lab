import json
from pathlib import Path
import subprocess
import sys
import shutil
import threading

from scripts.runtime.supervise_warm_task_recovery import (
    COMPLETE_FAILURE,
    COMPLETE_SUCCESS,
    HELDOUT_REQUIRED,
    SYNTHESIS_REQUIRED,
    controller_pid_from_file,
    job_phase,
    recovery_processes,
    write_state,
)


def _job(tmp_path):
    return {
        "cell_id": "gsm-qwen35-2b",
        "output_name": "warmfix_gsm-qwen35-2b_0714_r2",
        "heldout_output_json": str(tmp_path / "heldout.json"),
    }


def _latest_run(tmp_path, job, report_name=None):
    output_root = tmp_path / "outputs" / "generated" / job["output_name"]
    run_dir = output_root / "run-1"
    (run_dir / "results").mkdir(parents=True)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "latest_run.txt").write_text(str(run_dir))
    if report_name:
        (run_dir / "results" / report_name).write_text("{}")


def test_job_phase_restarts_only_nonterminal_synthesis(tmp_path):
    job = _job(tmp_path)

    assert job_phase(tmp_path, job) == SYNTHESIS_REQUIRED

    _latest_run(tmp_path, job, "failure_report.json")

    assert job_phase(tmp_path, job) == COMPLETE_FAILURE


def test_job_phase_resumes_heldout_without_rerunning_synthesis(tmp_path):
    job = _job(tmp_path)
    _latest_run(tmp_path, job, "success_report.json")

    assert job_phase(tmp_path, job) == HELDOUT_REQUIRED

    heldout = tmp_path / "heldout.json"
    heldout.write_text(json.dumps({"accuracy": 0.5}))

    assert job_phase(tmp_path, job) == COMPLETE_SUCCESS


def test_state_file_records_every_row_atomically(tmp_path):
    state_path = tmp_path / "state.json"
    jobs = [_job(tmp_path), {**_job(tmp_path), "cell_id": "other"}]
    phases = {
        "gsm-qwen35-2b": SYNTHESIS_REQUIRED,
        "other": COMPLETE_FAILURE,
    }

    write_state(state_path, jobs, phases, controller_pid=123)

    state = json.loads(state_path.read_text())
    assert state["version"] == 1
    assert state["controller_pid"] == 123
    assert state["jobs"]["gsm-qwen35-2b"]["phase"] == SYNTHESIS_REQUIRED
    assert state["jobs"]["other"]["phase"] == COMPLETE_FAILURE
    assert not state_path.with_suffix(".json.tmp").exists()


def test_focal_user_service_restarts_failures_and_kills_child_workers():
    unit = (
        Path(__file__).parents[2]
        / "deploy"
        / "focal"
        / "systemd"
        / "csd-warm-recovery.service"
    ).read_text()

    assert "supervise_warm_task_recovery.py" in unit
    assert "Restart=on-failure" in unit
    assert "KillMode=control-group" in unit
    assert "--retry-seconds 3600" in unit
    assert "AWS_BEARER_TOKEN_BEDROCK" not in unit


def test_stale_pid_file_does_not_adopt_an_unrelated_process(tmp_path):
    pid_file = tmp_path / "controller.pid"
    proc_root = tmp_path / "proc"
    process = proc_root / "123"
    process.mkdir(parents=True)
    pid_file.write_text("123\n")
    (process / "cmdline").write_bytes(b"python\0unrelated.py\0")

    expected_manifest = tmp_path / "manifest.json"
    assert controller_pid_from_file(pid_file, proc_root, expected_manifest) is None

    (process / "cmdline").write_bytes(
        b"python\0run_warm_task_recovery_queue.py\0--manifest\0/different.json\0"
    )

    assert controller_pid_from_file(pid_file, proc_root, expected_manifest) is None

    (process / "cmdline").write_bytes(
        b"python\0run_warm_task_recovery_queue.py\0--manifest\0"
        + str(expected_manifest).encode()
        + b"\0"
    )

    assert controller_pid_from_file(pid_file, proc_root, expected_manifest) == 123


def test_recovery_processes_include_vllm_descendants_but_not_siblings(tmp_path):
    proc_root = tmp_path / "proc"

    def process(pid, ppid, *arguments):
        directory = proc_root / str(pid)
        directory.mkdir(parents=True)
        (directory / "cmdline").write_bytes(
            b"\0".join(str(argument).encode() for argument in arguments) + b"\0"
        )
        (directory / "status").write_text(f"Name:\ttest\nPPid:\t{ppid}\n")

    process(
        100,
        1,
        "python",
        "-m",
        "synthesis.run_synthesis",
        "--output-name",
        "warmfix_gsm-qwen35-2b_0714_r2",
    )
    process(101, 100, "VLLM::EngineCore")
    process(102, 1, "VLLM::EngineCore")

    for pid in (100, 101, 102):
        (proc_root / str(pid) / "cwd").symlink_to(tmp_path)

    assert recovery_processes(proc_root, [_job(tmp_path)], expected_repo=tmp_path) == {100, 101}


def test_recovery_processes_reject_same_output_name_from_other_checkout(tmp_path):
    proc_root = tmp_path / "proc"
    expected_repo = tmp_path / "expected"
    other_repo = tmp_path / "other"
    expected_repo.mkdir()
    other_repo.mkdir()
    process = proc_root / "200"
    process.mkdir(parents=True)
    (process / "cmdline").write_bytes(
        b"python\0-m\0synthesis.run_synthesis\0--output-name\0"
        b"warmfix_gsm-qwen35-2b_0714_r2\0"
    )
    (process / "status").write_text("Name:\ttest\nPPid:\t1\n")
    (process / "cwd").symlink_to(other_repo)

    assert recovery_processes(
        proc_root, [_job(tmp_path)], expected_repo=expected_repo
    ) == set()


def test_supervisor_adopts_live_controller_then_finishes_pending_row(tmp_path):
    source = tmp_path / "source.log"
    history = tmp_path / "history.json"
    source.write_text("source")
    history.write_text("[]")
    job = {
        **_job(tmp_path),
        "last_clean_attempt": 10,
        "total_cap": 40,
        "memory_reservation_mib": 16384,
        "source_log": str(source),
        "history_file": str(history),
        "log_file": str(tmp_path / "worker.log"),
        "dataset": "gsm_symbolic",
        "eval_model": "Qwen/Qwen3.5-2B",
        "gpu_mem_util": 0.4,
        "heldout_sample_size": 49,
        "eval_max_steps": 900,
        "eval_max_seconds": 600,
        "heldout_split_name": "eval",
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([job]))
    fake_controller = tmp_path / "run_warm_task_recovery_queue.py"
    fake_controller.write_text(
        """import json, sys, time
from pathlib import Path
if '--manifest' not in sys.argv:
    time.sleep(0.2)
    raise SystemExit(0)
manifest = Path(sys.argv[sys.argv.index('--manifest') + 1])
repo = Path(sys.argv[sys.argv.index('--repo') + 1])
for job in json.loads(manifest.read_text()):
    run_dir = repo / 'outputs' / 'generated' / job['output_name'] / 'run-1'
    (run_dir / 'results').mkdir(parents=True, exist_ok=True)
    (run_dir.parent / 'latest_run.txt').write_text(str(run_dir))
    (run_dir / 'results' / 'failure_report.json').write_text('{}')
(repo / 'fake-controller-ran').write_text('yes')
"""
    )
    adopted = subprocess.Popen([sys.executable, str(fake_controller)])
    pid_file = tmp_path / "controller.pid"
    pid_file.write_text(f"{adopted.pid}\n")
    proc_root = tmp_path / "proc"
    adopted_proc = proc_root / str(adopted.pid)
    adopted_proc.mkdir(parents=True)
    (adopted_proc / "cmdline").write_bytes(
        f"{sys.executable}\0{fake_controller}\0".encode()
    )

    def remove_finished_process_record():
        adopted.wait()
        shutil.rmtree(adopted_proc)

    threading.Thread(target=remove_finished_process_record, daemon=True).start()
    state_file = tmp_path / "state.json"
    supervisor = Path(__file__).parents[2] / "scripts" / "runtime" / "supervise_warm_task_recovery.py"

    result = subprocess.run(
        [
            sys.executable,
            str(supervisor),
            "--repo", str(tmp_path),
            "--manifest", str(manifest),
            "--controller", str(fake_controller),
            "--resume-script", str(tmp_path / "resume.sh"),
            "--python", sys.executable,
            "--controller-lock", str(tmp_path / "controller.lock"),
            "--controller-pid-file", str(pid_file),
            "--state-file", str(state_file),
            "--pending-manifest", str(tmp_path / "pending.json"),
            "--poll-seconds", "0.05",
            "--retry-seconds", "0.1",
            "--proc-root", str(proc_root),
        ],
        cwd=Path(__file__).parents[2],
        capture_output=True,
        text=True,
        timeout=5,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "fake-controller-ran").read_text() == "yes"
    state = json.loads(state_file.read_text())
    assert state["controller_pid"] is None
    assert state["jobs"][job["cell_id"]]["phase"] == COMPLETE_FAILURE
