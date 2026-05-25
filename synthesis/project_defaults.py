from __future__ import annotations

import os
from pathlib import Path
from shutil import which


def repo_root() -> Path:
    """Repository root (parent of the ``synthesis/`` package)."""
    return Path(__file__).resolve().parent.parent


def default_logs_dir() -> Path:
    """Canonical directory for synthesis prompt/response logs."""
    raw = os.environ.get("CSD_LOGS_DIR")
    if raw:
        return Path(raw).expanduser()
    return repo_root() / "logs"


def synthesis_prompt_log_dir(output_name: str, run_id: str) -> Path:
    """Per-run prompt log directory under :func:`default_logs_dir`."""
    return default_logs_dir() / f"{output_name}_{run_id}"


def default_repo_path(env_var: str, fallback_name: str) -> Path:
    raw = os.environ.get(env_var)
    if raw:
        return Path(raw).expanduser()
    return Path.home() / fallback_name


def default_crane_repo() -> Path:
    return default_repo_path("CRANE_REPO", "CRANE")


def default_itergen_repo() -> Path:
    return default_repo_path("ITERGEN_REPO", "itergen")



def default_gsm_source_dir() -> Path:
    env_gsm = os.environ.get("CRANE_GSM_SYMBOLIC_DIR")
    if env_gsm:
        p = Path(env_gsm).expanduser()
        if p.exists():
            return p
    repo_root = Path(__file__).resolve().parent.parent
    for candidate in (
        repo_root / "legacy" / "CRANE" / "src" / "gsm_symbolic",
        default_crane_repo() / "src" / "gsm_symbolic",
        default_crane_repo() / "gsm_symbolic",
    ):
        if candidate.exists():
            return candidate
    return default_crane_repo() / "src" / "gsm_symbolic"


def default_dafny_path() -> str:
    """Resolve the Dafny executable for verify/compile subprocesses.

    Precedence: ``DAFNY_PATH`` env, then repo ``dafny/dafny``, then ``dafny`` on
    ``PATH``, then ``~/.dotnet/tools/dafny``.
    """
    env_dafny = os.environ.get("DAFNY_PATH")
    if env_dafny:
        return env_dafny
    repo_dafny = repo_root() / "dafny" / "dafny"
    if repo_dafny.is_file():
        return str(repo_dafny)
    path_dafny = which("dafny")
    if path_dafny:
        return path_dafny
    return str(Path.home() / ".dotnet" / "tools" / "dafny")


def default_spider_data_dir() -> Path:
    return Path(os.environ.get("SPIDER_DATA_DIR", str(Path.home() / "spider_data" / "spider_data"))).expanduser()
