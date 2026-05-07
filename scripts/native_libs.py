"""Native library setup for Conda/venv entry points."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def ensure_env_lib_first() -> None:
    """Re-exec once with the current Python env's lib directory first."""
    python_lib = Path(sys.executable).resolve().parent.parent / "lib"
    if not python_lib.exists():
        return

    python_lib_str = str(python_lib)
    current = os.environ.get("LD_LIBRARY_PATH", "")
    current_parts = [part for part in current.split(":") if part]
    if current_parts and current_parts[0] == python_lib_str:
        return
    if os.environ.get("_VAS_NATIVE_LIBS_REEXECED") == "1":
        return

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(
        [python_lib_str, *[part for part in current_parts if part != python_lib_str]]
    )
    env["_VAS_NATIVE_LIBS_REEXECED"] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)
