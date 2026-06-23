"""Resolve shared cache and output directories for CLI entrypoints."""

from __future__ import annotations

import os
from pathlib import Path

from synthesis.project_defaults import repo_root


def default_cache_root() -> Path:
    return repo_root() / "cache"


def default_outputs_root() -> Path:
    return repo_root() / "outputs"


def ensure_repo_cache_env() -> Path:
    """Point HF + SynCode pickles at ``CSD_CACHE_ROOT`` (default: ``<repo>/cache``).

    Legacy CRANE/IterGen historically defaulted to cwd-relative ``iter_cache/``,
    duplicating multi-GB model snapshots. This helper keeps Hugging Face checkpoints
    and ``mask_stores/`` / ``parsers/`` together under one root.
    """
    cache_root = Path(
        os.environ.get("CSD_CACHE_ROOT", str(default_cache_root()))
    ).expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    root_s = str(cache_root)

    os.environ.setdefault("CSD_CACHE_ROOT", root_s)
    os.environ.setdefault("HF_HOME", root_s)
    os.environ.setdefault("HF_CACHE", root_s)
    os.environ.setdefault("TRANSFORMERS_CACHE", root_s)

    syn_existing = os.environ.get("SYNCODE_CACHE") or os.environ.get("ITER_SYNCODE_CACHE")
    if syn_existing:
        syn = syn_existing if syn_existing.endswith(os.sep) else syn_existing + os.sep
        os.environ.setdefault("SYNCODE_CACHE", syn)
        os.environ.setdefault("ITER_SYNCODE_CACHE", syn)
    else:
        syn = root_s if root_s.endswith(os.sep) else root_s + os.sep
        os.environ.setdefault("SYNCODE_CACHE", syn)
        os.environ.setdefault("ITER_SYNCODE_CACHE", syn)

    return cache_root


def ensure_repo_outputs_env() -> Path:
    """Point run artifacts at ``CSD_OUTPUTS_ROOT`` (default: ``<repo>/outputs``).

    Sets ``CSD_OUTPUT_DIR``, ``CSD_BASELINE_OUTPUT_DIR``, ``CSD_ABLATION_OUTPUT_DIR``,
    and ``CSD_GPU3_RETRY_QUEUE`` unless already set. Does **not** relocate ``logs/``.
    """
    outputs_root = Path(
        os.environ.get("CSD_OUTPUTS_ROOT", str(default_outputs_root()))
    ).expanduser().resolve()
    outputs_root.mkdir(parents=True, exist_ok=True)
    root_s = str(outputs_root)

    os.environ.setdefault("CSD_OUTPUTS_ROOT", root_s)
    os.environ.setdefault("CSD_OUTPUT_DIR", str(outputs_root / "generated"))
    os.environ.setdefault("CSD_BASELINE_OUTPUT_DIR", str(outputs_root / "baselines"))
    os.environ.setdefault("CSD_ABLATION_OUTPUT_DIR", str(outputs_root / "ablations"))
    os.environ.setdefault(
        "CSD_GPU3_RETRY_QUEUE",
        str(outputs_root / "gpu3_retry_queue.jsonl"),
    )

    for name in ("generated", "baselines", "ablations"):
        (outputs_root / name).mkdir(parents=True, exist_ok=True)

    return outputs_root


def ensure_shared_storage_env() -> tuple[Path, Path]:
    """Apply cache + output path defaults (not logs)."""
    return ensure_repo_cache_env(), ensure_repo_outputs_env()


# Backward-compatible alias used by older imports.
_ensure_repo_cache_env = ensure_repo_cache_env
