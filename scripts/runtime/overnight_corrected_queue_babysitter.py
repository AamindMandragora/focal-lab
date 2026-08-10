#!/usr/bin/env python3
"""Overnight babysitter for full-baseline-corrected-20260805.

Exit predicates this process cares about locally:
1) every recoverable missing_csd cell gets heldout recovery attempted
2) keep polling for newly finished cells that land as missing_csd
3) never touch GPU 1
4) never kill the live cold controller; only recover around it

Stops when --until-idle and there are no recoverable missing_csd cells and no
active synthesis/reeval children for this campaign for --idle-polls consecutive polls.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from pathlib import Path

from scripts.runtime.run_cold_synthesis_queue import (
    compiled_csd,
    gpu_memory_snapshot,
    run_job,
)

REPO = Path("/home/aadivyar/csd-generation-worktrees/full-baseline-campaign-20260803")
MANIFEST = REPO / "saved-results/2026-08-05-corrected-full-baseline-cold-manifest.json"
STATE_DIR = REPO / ".context/full-baseline-corrected-20260805-cold-state"
PYTHON = Path("/apps/conda/aadivyar/envs/csd/bin/python")
ALLOWED = (0, 2, 3)
CONTROLLER_PID_FILE = REPO / ".context/full-baseline-corrected-20260805-cold.pid"
STATUS_PATH = REPO / "saved-results/2026-08-05-overnight-babysitter-status.json"
LOG = REPO / "logs/full-baseline-corrected-20260805-overnight-babysitter.log"
POLL_SECONDS = 60


def load_jobs() -> list[dict]:
    return json.loads(MANIFEST.read_text())["jobs"]


def write_status(payload: dict) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATUS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(STATUS_PATH)


def controller_alive() -> bool:
    try:
        pid = int(CONTROLLER_PID_FILE.read_text().strip())
    except Exception:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def campaign_children() -> list[str]:
    out = subprocess.check_output(["ps", "-u", "aadivyar", "-o", "pid=,cmd="], text=True)
    lines = []
    for line in out.splitlines():
        if "overnight_corrected_queue_babysitter" in line:
            continue
        if "recover_missing_csd_heldout" in line:
            continue
        if "campaign_health_babysitter" in line:
            continue
        if "coldq_corrected_20260805" in line or "full_baseline_corrected_20260805" in line:
            lines.append(line)
        elif "reevaluate_compiled_csd" in line and "full_baseline_corrected_20260805" in line:
            lines.append(line)
        elif "synthesis.run_synthesis" in line:
            # Controller children often omit the coldq token from argv.
            lines.append(line)
    return lines


def gpu1_has_compute() -> bool:
    """True only if *our* user has a compute process on GPU1.

    Other users on GPU1 must not block recovery on GPUs 0/2/3.
    """
    try:
        apps = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                "1",
                "--query-compute-apps=pid,process_name",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
    except Exception:
        return False
    if not apps:
        return False
    for line in apps.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if not parts or not parts[0]:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        try:
            user = subprocess.check_output(["ps", "-p", str(pid), "-o", "user="], text=True).strip()
        except Exception:
            continue
        if user == "aadivyar":
            return True
    return False


def one_shot_recovery_running() -> bool:
    try:
        out = subprocess.check_output(["pgrep", "-af", "recover_missing_csd_heldout.py"], text=True)
    except subprocess.CalledProcessError:
        return False
    return "recover_missing_csd_heldout.py" in out


def recoverable_jobs(jobs: list[dict]) -> list[dict]:
    out = []
    for job in jobs:
        cell = job["cell_id"]
        state_path = STATE_DIR / f"{cell}.json"
        if not state_path.is_file():
            continue
        state = json.loads(state_path.read_text())
        if state.get("reason") != "missing_csd":
            continue
        if state.get("status") in {"complete_success", "complete_loss", "complete_failure"}:
            continue
        csd = compiled_csd(
            REPO,
            job["output_name"],
            min_accuracy=float(job["min_accuracy"]),
            min_syntax_rate=float(job["min_syntax_rate"]),
            job=job,
        )
        if csd is None:
            logging.warning("unrecoverable missing_csd cell=%s", cell)
            continue
        heldout = Path(str(job["heldout_output_json"]))
        if not heldout.is_absolute():
            heldout = REPO / heldout
        if heldout.is_file():
            # heldout exists but state not updated; leave for a later reconcile pass
            logging.info("heldout exists but state still missing_csd cell=%s", cell)
        out.append(job)
    return out


def pick_gpu(job: dict) -> int | None:
    snap = gpu_memory_snapshot("nvidia-smi")
    need = int(job.get("memory_reservation_mib") or 16000)
    ranked = sorted(
        ((int((snap.get(g) or {}).get("free_mib") or 0), g) for g in ALLOWED),
        reverse=True,
    )
    for free, gpu in ranked:
        if free >= need + 2048:
            return gpu
    free, gpu = ranked[0]
    if free >= 12000:
        return gpu
    return None


def recover_one(job: dict) -> int:
    cell = job["cell_id"]
    gpu = pick_gpu(job)
    if gpu is None:
        logging.info("no GPU capacity for cell=%s", cell)
        return -1
    logging.info("recover heldout cell=%s gpu=%s", cell, gpu)
    status = run_job(dict(job), (gpu,), repo=REPO, python=PYTHON, state_dir=STATE_DIR)
    state = json.loads((STATE_DIR / f"{cell}.json").read_text())
    logging.info("recover finish cell=%s status=%s state=%s", cell, status, state)
    return status


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(LOG), logging.StreamHandler()],
    )
    idle_polls = 0
    jobs = load_jobs()
    logging.info("overnight babysitter start controller_alive=%s", controller_alive())
    while True:
        if gpu1_has_compute():
            logging.error("GPU1 has compute processes; refusing to launch recovery")
        if one_shot_recovery_running():
            logging.info("one-shot recovery still running; babysitter waits")
            write_status(
                {
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "controller_alive": controller_alive(),
                    "waiting_on_one_shot_recovery": True,
                    "gpu1_compute": gpu1_has_compute(),
                }
            )
            time.sleep(POLL_SECONDS)
            continue

        pending = recoverable_jobs(jobs)
        children = campaign_children()
        status = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "controller_alive": controller_alive(),
            "recoverable_missing_csd": [j["cell_id"] for j in pending],
            "campaign_children": len(children),
            "gpu1_compute": gpu1_has_compute(),
            "heldout_files": sorted(
                p.name
                for p in (REPO / "outputs/reeval/full_baseline_corrected_20260805").glob("*.json")
            )
            if (REPO / "outputs/reeval/full_baseline_corrected_20260805").exists()
            else [],
        }
        write_status(status)
        logging.info("status %s", json.dumps(status))

        if pending and not gpu1_has_compute():
            # one cell per loop to keep GPU contention small
            code = recover_one(pending[0])
            idle_polls = 0
            if code == -1:
                time.sleep(POLL_SECONDS)
            continue

        if not pending and not children and not controller_alive():
            idle_polls += 1
            logging.info("idle poll %s/5", idle_polls)
            if idle_polls >= 5:
                logging.info("overnight babysitter exit: idle")
                write_status({**status, "stopped": "idle"})
                return 0
        else:
            idle_polls = 0
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
