#!/usr/bin/env python3
"""Campaign health babysitter for full-baseline-corrected-20260805.

Watches live campaign health and takes safe automatic repairs:
1) restart missing_csd overnight babysitter if it dies
2) restart cold controller if it dies while cells remain pending
3) never place campaign work on GPU1; alert if our user appears on GPU1
4) alert on progress stalls (synthesis alive, progress_report mtime stale)
5) recover missing_csd remains delegated to overnight_corrected_queue_babysitter.py

Never kills live synthesis mid-flight. Never touches other users' GPU1 jobs.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from pathlib import Path

REPO = Path("/home/aadivyar/csd-generation-worktrees/full-baseline-campaign-20260803")
PYTHON = Path("/apps/conda/aadivyar/envs/csd/bin/python")
MANIFEST = REPO / "saved-results/2026-08-05-corrected-full-baseline-cold-manifest.json"
APPROVAL = REPO / "saved-results/2026-08-05-corrected-full-baseline-launch-approval.json"
STATE_DIR = REPO / ".context/full-baseline-corrected-20260805-cold-state"
CONTROLLER_PID = REPO / ".context/full-baseline-corrected-20260805-cold.pid"
CONTROLLER_LOCK = REPO / ".context/full-baseline-corrected-20260805-cold.lock"
MISSING_CSD_BABYSITTER_PID = REPO / ".context/full-baseline-corrected-20260805-overnight-babysitter.pid"
HEALTH_PID = REPO / ".context/full-baseline-corrected-20260805-health-babysitter.pid"
STATUS = REPO / "saved-results/2026-08-05-campaign-health-babysitter-status.json"
LOG = REPO / "logs/full-baseline-corrected-20260805-campaign-health-babysitter.log"
MISSING_CSD_SCRIPT = REPO / "scripts/runtime/overnight_corrected_queue_babysitter.py"
POLL_SECONDS = 60
PROGRESS_STALE_SECONDS = 4 * 3600
OUR_USER = "aadivyar"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def read_pid(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except Exception:
        return None


def ps_lines() -> list[str]:
    return subprocess.check_output(["ps", "-u", OUR_USER, "-o", "pid=,ppid=,etime=,cmd="], text=True).splitlines()


def controller_alive() -> bool:
    return alive(read_pid(CONTROLLER_PID))


def missing_csd_babysitter_alive() -> bool:
    return alive(read_pid(MISSING_CSD_BABYSITTER_PID))


def synthesis_procs() -> list[dict]:
    out = []
    for line in ps_lines():
        if "synthesis.run_synthesis" not in line:
            continue
        parts = line.split(None, 3)
        if len(parts) < 4:
            continue
        out.append({"pid": int(parts[0]), "ppid": int(parts[1]), "etime": parts[2], "cmd": parts[3][:180]})
    return out


def gpu1_our_compute() -> list[dict]:
    """Return compute apps on GPU1 owned by our user only."""
    try:
        apps = subprocess.check_output(
            ["nvidia-smi", "-i", "1", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
            text=True,
        ).strip()
    except Exception as exc:
        logging.warning("nvidia-smi gpu1 failed: %s", exc)
        return []
    found = []
    if not apps:
        return found
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
        if user == OUR_USER:
            found.append(
                {
                    "pid": pid,
                    "process": parts[1] if len(parts) > 1 else "?",
                    "used_memory": parts[2] if len(parts) > 2 else "?",
                    "user": user,
                }
            )
    return found


def pending_cells() -> list[str]:
    jobs = json.loads(MANIFEST.read_text())["jobs"]
    done_ish = set()
    for p in STATE_DIR.glob("*.json"):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        st = d.get("status")
        if st in ("complete_success", "complete_loss", "success", "done"):
            done_ish.add(d.get("cell_id") or p.stem)
    return [j["cell_id"] for j in jobs if j["cell_id"] not in done_ish]


def missing_csd_cells() -> list[str]:
    out = []
    for p in STATE_DIR.glob("*.json"):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if d.get("reason") == "missing_csd":
            out.append(d.get("cell_id") or p.stem)
    return out


def progress_staleness() -> list[dict]:
    alerts = []
    gen = REPO / "outputs" / "generated"
    if not gen.exists():
        return alerts
    now = time.time()
    synth = synthesis_procs()
    if not synth:
        return alerts
    newest = None
    newest_mtime = 0.0
    for report in gen.glob("coldq_corrected_20260805_*/**/progress_report.json"):
        try:
            mtime = report.stat().st_mtime
        except OSError:
            continue
        if mtime > newest_mtime:
            newest_mtime = mtime
            newest = report
    if newest is None:
        for proc in synth:
            alerts.append({"pid": proc["pid"], "issue": "no_progress_report_found", "etime": proc["etime"]})
        return alerts
    age = now - newest_mtime
    if age > PROGRESS_STALE_SECONDS:
        for proc in synth:
            alerts.append(
                {
                    "pid": proc["pid"],
                    "issue": "progress_report_stale",
                    "age_seconds": int(age),
                    "path": str(newest.relative_to(REPO)),
                    "etime": proc["etime"],
                }
            )
    return alerts


def _repo_env() -> dict:
    env = os.environ.copy()
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(REPO) if not prev else f"{REPO}{os.pathsep}{prev}"
    return env


def start_missing_csd_babysitter() -> int:
    logging.warning("restarting missing_csd babysitter")
    log = open(REPO / "logs/full-baseline-corrected-20260805-overnight-babysitter.log", "a")
    proc = subprocess.Popen(
        [str(PYTHON), str(MISSING_CSD_SCRIPT)],
        cwd=str(REPO),
        stdout=log,
        stderr=log,
        start_new_session=True,
        env=_repo_env(),
    )
    MISSING_CSD_BABYSITTER_PID.write_text(str(proc.pid) + "\n")
    logging.info("missing_csd babysitter pid=%s", proc.pid)
    return proc.pid


def start_controller() -> int:
    logging.error("restarting cold controller (pending cells remain)")
    if CONTROLLER_LOCK.exists() and not controller_alive():
        try:
            CONTROLLER_LOCK.unlink()
        except OSError:
            pass
    log = open(REPO / "logs/full-baseline-corrected-20260805-cold-controller.log", "a")
    cmd = [
        str(PYTHON),
        "-m",
        "scripts.runtime.run_cold_synthesis_queue",
        "--repo",
        str(REPO),
        "--manifest",
        str(MANIFEST),
        "--corrected-approval",
        str(APPROVAL),
        "--python",
        str(PYTHON),
        "--lock-file",
        str(CONTROLLER_LOCK),
        "--state-dir",
        str(STATE_DIR),
        "--campaign-profile",
        "full-baseline-corrected-20260805",
        "--gpus",
        "0,2,3",
        "--poll-seconds",
        "30",
        "--repair-attestation",
        str(REPO / "saved-results/2026-08-06-greedy-hillclimb-repair-attestation.json"),
    ]
    proc = subprocess.Popen(cmd, cwd=str(REPO), stdout=log, stderr=log, start_new_session=True, env=_repo_env())
    CONTROLLER_PID.write_text(str(proc.pid) + "\n")
    logging.info("controller restarted pid=%s", proc.pid)
    return proc.pid


def main() -> int:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(LOG), logging.StreamHandler()],
    )
    HEALTH_PID.write_text(str(os.getpid()) + "\n")
    logging.info("campaign health babysitter start pid=%s", os.getpid())

    while True:
        alerts: list[dict] = []
        actions: list[str] = []

        our_gpu1 = gpu1_our_compute()
        if our_gpu1:
            alerts.append({"severity": "our_process_on_gpu1", "apps": our_gpu1})
            logging.error("OUR process on GPU1: %s", our_gpu1)

        if not missing_csd_babysitter_alive():
            alerts.append({"severity": "missing_csd_babysitter_dead"})
            try:
                start_missing_csd_babysitter()
                actions.append("restarted_missing_csd_babysitter")
            except Exception as exc:
                alerts.append({"severity": "missing_csd_babysitter_restart_failed", "error": str(exc)})

        pending = pending_cells()
        if not controller_alive() and pending:
            alerts.append({"severity": "controller_dead_with_pending", "pending": pending})
            try:
                start_controller()
                actions.append("restarted_controller")
            except Exception as exc:
                alerts.append({"severity": "controller_restart_failed", "error": str(exc)})

        for stall in progress_staleness():
            alerts.append({"severity": "progress_stall", **stall})
            logging.warning("progress stall %s", stall)

        missing = missing_csd_cells()
        if missing:
            alerts.append({"severity": "missing_csd_present", "cells": missing})

        synth = synthesis_procs()
        status = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "health_pid": os.getpid(),
            "controller_alive": controller_alive(),
            "controller_pid": read_pid(CONTROLLER_PID),
            "missing_csd_babysitter_alive": missing_csd_babysitter_alive(),
            "missing_csd_babysitter_pid": read_pid(MISSING_CSD_BABYSITTER_PID),
            "pending_cells": pending,
            "pending_n": len(pending),
            "missing_csd": missing,
            "synthesis": synth,
            "our_gpu1_compute": our_gpu1,
            "alerts": alerts,
            "actions": actions,
        }
        write_json(STATUS, status)
        logging.info(
            "health controller=%s babysitter=%s pending=%s synth=%s alerts=%s actions=%s",
            status["controller_alive"],
            status["missing_csd_babysitter_alive"],
            status["pending_n"],
            len(synth),
            len(alerts),
            actions,
        )
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
