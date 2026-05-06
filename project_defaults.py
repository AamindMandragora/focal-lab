from __future__ import annotations

import os
from pathlib import Path


def default_repo_path(env_var: str, fallback_name: str) -> Path:
    raw = os.environ.get(env_var)
    if raw:
        return Path(raw).expanduser()
    return Path.home() / fallback_name


def default_crane_repo() -> Path:
    return default_repo_path("CRANE_REPO", "CRANE")


def default_itergen_repo() -> Path:
    return default_repo_path("ITERGEN_REPO", "itergen")


def default_cars_repo() -> Path:
    return default_repo_path("CARS_REPO", "cars")


def default_gsm_source_dir() -> Path:
    env_gsm = os.environ.get("CRANE_GSM_SYMBOLIC_DIR")
    if env_gsm:
        p = Path(env_gsm).expanduser()
        if p.exists():
            return p
    crane_root = default_crane_repo()
    for candidate in (crane_root / "src" / "gsm_symbolic", crane_root / "gsm_symbolic"):
        if candidate.exists():
            return candidate
    return crane_root / "src" / "gsm_symbolic"


def default_dafny_path() -> str:
    return os.environ.get("DAFNY_PATH", str(Path.home() / ".dotnet" / "tools" / "dafny"))


def default_spider_data_dir() -> Path:
    return Path(os.environ.get("SPIDER_DATA_DIR", str(Path.home() / "spider_data" / "spider_data"))).expanduser()
