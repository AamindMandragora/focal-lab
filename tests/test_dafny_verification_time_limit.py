"""Dafny verify and build must share the same 10-minute per-batch time limit."""

from synthesis.run_constants import (
    DAFNY_COMPILE_PROCESS_TIMEOUT_SECONDS,
    DAFNY_VERIFICATION_TIME_LIMIT_SECONDS,
    DAFNY_VERIFY_PROCESS_TIMEOUT_SECONDS,
)
from synthesis.verify.tooling import build_default_compiler, build_default_verifier


def test_dafny_verification_time_limit_is_ten_minutes():
    assert DAFNY_VERIFICATION_TIME_LIMIT_SECONDS == 600


def test_dafny_process_timeouts_cover_ten_minute_batch_limit():
    assert DAFNY_VERIFY_PROCESS_TIMEOUT_SECONDS >= DAFNY_VERIFICATION_TIME_LIMIT_SECONDS
    assert DAFNY_COMPILE_PROCESS_TIMEOUT_SECONDS >= DAFNY_VERIFICATION_TIME_LIMIT_SECONDS


def test_default_verifier_and_compiler_share_verification_time_limit(tmp_path):
    expected = ["--verification-time-limit", "600"]
    verifier = build_default_verifier(dafny_path="dafny")
    compiler = build_default_compiler(dafny_path="dafny", output_dir=tmp_path)

    assert verifier.extra_args == expected
    assert compiler.extra_args == expected
    assert verifier.timeout == DAFNY_VERIFY_PROCESS_TIMEOUT_SECONDS
    assert compiler.timeout == DAFNY_COMPILE_PROCESS_TIMEOUT_SECONDS
