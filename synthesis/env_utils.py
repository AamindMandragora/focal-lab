"""Environment file loading shared by CLI entrypoints."""

from __future__ import annotations

import os
import shlex
from pathlib import Path


def parse_env_value(raw: str) -> str:
    raw = raw.strip()
    if not raw:
        return ""
    try:
        parsed = shlex.split(raw, posix=True)
    except ValueError:
        return raw.strip("\"'")
    if not parsed:
        return ""
    return parsed[0]


def load_env_file(path: Path) -> None:
    """Load KEY=VALUE lines from a dotenv-style file into ``os.environ``."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].strip()
        if "=" not in stripped:
            continue
        key, raw_value = stripped.split("=", 1)
        key = key.strip()
        if key:
            os.environ[key] = parse_env_value(raw_value)
