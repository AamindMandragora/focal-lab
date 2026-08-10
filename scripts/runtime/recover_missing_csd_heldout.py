#!/usr/bin/env python3
"""Recover heldout for cells falsely marked missing_csd after successful synthesis."""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

from scripts.runtime.run_cold_synthesis_queue import (
    compiled_csd,
    gpu_memory_snapshot,
    run_job,
    stamp_job_commit_from_report,
)

REPO = Path("/home/aadivyar/csd-generation-worktrees/full-baseline-campaign-20260803")
MANIFEST = REPO / "saved-results/2026-08-05-corrected-full-baseline-cold-manifest.json"
STATE_DIR = REPO / ".context/full-baseline-corrected-20260805-cold-state"
PYTHON = Path("/apps/conda/aadivyar/envs/csd/bin/python")
ALLOWED = (0, 2, 3)
LOG = REPO / "logs/full-baseline-corrected-20260805-heldout-recovery.log"


def cells_needing_recovery(jobs: list[dict]) -> list[dict]:
    out = []
    for job in jobs:
        cell = job["cell_id"]
        state_path = STATE_DIR / f"{cell}.json"
        if not state_path.is_file():
            continue
        state = json.loads(state_path.read_text())
        if state.get("reason") != "missing_csd":
            continue
        csd = compiled_csd(
            REPO,
            job["output_name"],
            min_accuracy=float(job["min_accuracy"]),
            min_syntax_rate=float(job["min_syntax_rate"]),
            job=job,
        )
        if csd is None:
            logging.warning("skip %s: still no compiled csd", cell)
            continue
        heldout = Path(job["heldout_output_json"])
        if not heldout.is_absolute():
            heldout = REPO / heldout
        if heldout.is_file():
            logging.warning("skip %s: heldout already exists %s", cell, heldout)
            continue
        out.append(job)
    return out


def pick_gpu(job: dict) -> int | None:
    snap = gpu_memory_snapshot("nvidia-smi")
    need = int(job.get("memory_reservation_mib") or 16000)
    # free_mib key from snapshot
    candidates = []
    for gpu in ALLOWED:
        info = snap.get(gpu) or {}
        free = int(info.get("free_mib") or 0)
        candidates.append((free, gpu))
    candidates.sort(reverse=True)
    for free, gpu in candidates:
        if free >= need + 2048:
            return gpu
    # fallback: largest free if at least 12GB
    free, gpu = candidates[0]
    if free >= 12000:
        return gpu
    return None


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.FileHandler(LOG),
            logging.StreamHandler(),
        ],
    )
    payload = json.loads(MANIFEST.read_text())
    jobs = payload["jobs"]
    # Stamp like main() would for report matching on unpinned manifests:
    # leave null; compiled_csd uses our fix. Then stamp from report before heldout inside run_job.
    pending = cells_needing_recovery(jobs)
    logging.info("recovery pending=%d cells=%s", len(pending), [j["cell_id"] for j in pending])
    while pending:
        job = pending[0]
        cell = job["cell_id"]
        gpu = pick_gpu(job)
        if gpu is None:
            logging.info("wait for GPU memory cell=%s", cell)
            time.sleep(30)
            continue
        logging.info("recover heldout cell=%s gpu=%s", cell, gpu)
        # Clear error state so run_job can rewrite; run_job itself overwrites.
        status = run_job(
            dict(job),
            (gpu,),
            repo=REPO,
            python=PYTHON,
            state_dir=STATE_DIR,
        )
        logging.info("recover finish cell=%s status=%s", cell, status)
        state = json.loads((STATE_DIR / f"{cell}.json").read_text())
        logging.info("recover state cell=%s payload=%s", cell, state)
        pending.pop(0)
    logging.info("recovery complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
