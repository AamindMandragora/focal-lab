"""Shared constructors for Dafny verify/build with matching time limits."""

from __future__ import annotations

from pathlib import Path

from synthesis.run_constants import (
    DAFNY_COMPILE_PROCESS_TIMEOUT_SECONDS,
    DAFNY_VERIFICATION_TIME_LIMIT_SECONDS,
    DAFNY_VERIFY_PROCESS_TIMEOUT_SECONDS,
)
from synthesis.verify.compiler import DafnyCompiler
from synthesis.verify.verifier import DafnyVerifier


def verification_time_limit_args() -> list[str]:
    return ["--verification-time-limit", str(DAFNY_VERIFICATION_TIME_LIMIT_SECONDS)]


def build_default_verifier(*, dafny_path: str = "dafny") -> DafnyVerifier:
    return DafnyVerifier(
        dafny_path=dafny_path,
        timeout=DAFNY_VERIFY_PROCESS_TIMEOUT_SECONDS,
        extra_args=verification_time_limit_args(),
    )


def build_default_compiler(
    *,
    dafny_path: str = "dafny",
    output_dir: str | Path | None = None,
) -> DafnyCompiler:
    kwargs: dict = {
        "dafny_path": dafny_path,
        "timeout": DAFNY_COMPILE_PROCESS_TIMEOUT_SECONDS,
        "extra_args": verification_time_limit_args(),
    }
    if output_dir is not None:
        kwargs["output_dir"] = output_dir
    return DafnyCompiler(**kwargs)
