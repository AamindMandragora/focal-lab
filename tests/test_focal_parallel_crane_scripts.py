from pathlib import Path


SCRIPT_DIR = Path(__file__).parents[1] / "scripts" / "focal_parallel_crane"


def test_launcher_claims_output_and_records_process_identity() -> None:
    launch = (SCRIPT_DIR / "launch.sh").read_text()

    assert "set -euo pipefail" in launch
    assert 'claim_dir="${output_json}.running"' in launch
    assert 'mkdir "$claim_dir"' in launch
    assert "setsid" in launch
    assert "/proc/$worker_pid/stat" in launch
    assert "worker.meta" in launch
    assert "process_group_alive" in launch
    assert "retaining_claim" in launch
    assert 'candidate_pid=${worker_pid:-$!}' in launch
    assert "handle_signal()" in launch


def test_guard_verifies_identity_and_fails_safe_without_telemetry() -> None:
    guard = (SCRIPT_DIR / "guard.sh").read_text()

    assert "pid_wait_limit_seconds=60" in guard
    assert "shutdown_wait_limit_seconds=30" in guard
    assert "/proc/$worker_pid/cmdline" in guard
    assert "/proc/$worker_pid/stat" in guard
    assert "verified_identity=1" in guard
    assert "process_identity_changed" in guard
    assert "memory_query_failed" in guard
    assert "shutdown_wait_timeout" in guard
    assert "worker_group_alive" in guard
    assert 'kill -TERM -- "-$worker_pgid"' in guard
    assert 'if ! worker_group_alive; then' in guard
    assert 'rmdir "$claim_dir"' in guard
