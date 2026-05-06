#!/usr/bin/env python3
"""Small helpers for choosing GPUs for original-framework subprocesses."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class GpuInfo:
    index: str
    free_mib: int
    used_mib: int


def _parse_gpu_set(raw: str | None) -> set[str]:
    if not raw:
        return set()
    return {part.strip() for part in raw.split(",") if part.strip()}


def visible_device_count(raw: str | None) -> int:
    values = _parse_gpu_set(raw)
    return max(1, len(values))


def query_gpus() -> list[GpuInfo]:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return []

    gpus: list[GpuInfo] = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            gpus.append(GpuInfo(index=parts[0], free_mib=int(parts[1]), used_mib=int(parts[2])))
        except ValueError:
            continue
    return gpus


def select_cuda_visible_devices(
    *,
    requested: str | None = None,
    count: int = 1,
    min_free_mib: int | None = None,
    avoid: str | None = None,
) -> str:
    """Return a CUDA_VISIBLE_DEVICES string.

    `requested` can be an explicit comma-separated list or `auto`. Auto chooses
    the GPUs with the most free memory at the moment the subprocess is launched.
    """
    requested = (requested or "auto").strip()
    if requested and requested.lower() != "auto":
        return requested

    gpus = query_gpus()
    if not gpus:
        return ""

    avoid_set = _parse_gpu_set(avoid or os.environ.get("GPU_AVOID_DEVICES"))
    min_free = int(min_free_mib if min_free_mib is not None else os.environ.get("GPU_MIN_FREE_MIB", "12000"))
    candidates = [gpu for gpu in gpus if gpu.index not in avoid_set]
    candidates.sort(key=lambda gpu: (gpu.free_mib, -gpu.used_mib), reverse=True)
    enough = [gpu for gpu in candidates if gpu.free_mib >= min_free]
    selected = enough[:count] if len(enough) >= count else candidates[:count]
    return ",".join(gpu.index for gpu in selected)


def cuda_env(
    *,
    requested: str | None = None,
    count: int = 1,
    min_free_mib: int | None = None,
    avoid: str | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    selected = select_cuda_visible_devices(
        requested=requested,
        count=count,
        min_free_mib=min_free_mib,
        avoid=avoid,
    )
    if selected:
        env["CUDA_VISIBLE_DEVICES"] = selected
        print(f"[gpu-select] CUDA_VISIBLE_DEVICES={selected}", flush=True)
    return env
