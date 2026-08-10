#!/usr/bin/env python3
"""Pause corrected cold-queue data collection and hold GPUs 0,2,3 in memory.

Used when Claude hits a usage/session limit (exit 76). Stops authoring and
dispatch, keeps A100s occupied so other users cannot claim them, and leaves a
pause flag the Cursor babysit loop must honor until a human resumes.

Callers (manual / babysit loop, not imported):
  python .context/pause_queue_hold_gpus.py --reason claude_usage_limit --evidence '...'
  python .context/pause_queue_hold_gpus.py --check-hold
Documented in .audit/overnight-loop-prompt.txt after arming.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PAUSE_FLAG = REPO / ".context" / "full-baseline-corrected-20260805-usage-limit-pause.json"
HOLD_PID = REPO / ".context" / "full-baseline-corrected-20260805-gpu-hold.pid"
LOG = REPO / "logs" / "full-baseline-corrected-20260805-usage-limit-pause.log"

# Match campaign GPU scope. Never touch GPU 1.
HOLD_GPUS = (0, 2, 3)

# Fixed reservation after our synth dies. Do NOT use pre-kill nvidia-smi used
# memory as the target: that includes the processes we are about to free and
# can OOM the holder.
DEFAULT_MIB = {0: 24000, 2: 10000, 3: 14000}
MIN_HOLD_FRACTION = 0.70

KILL_PATTERNS = (
    "scripts.runtime.run_cold_synthesis_queue",
    "synthesis.run_synthesis",
    "scripts.runtime.campaign_health_babysitter",
    "scripts.runtime.overnight_corrected_queue_babysitter",
    "/home/aadivyar/.local/bin/claude --print",
    "reevaluate_compiled_csd",
)


def alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _pgrep(pattern: str) -> list[int]:
    try:
        out = subprocess.check_output(["pgrep", "-u", "aadivyar", "-f", pattern], text=True)
    except subprocess.CalledProcessError:
        return []
    return [int(x) for x in out.split() if x.strip().isdigit()]


def _children(pid: int) -> list[int]:
    try:
        out = subprocess.check_output(["pgrep", "-P", str(pid)], text=True)
    except subprocess.CalledProcessError:
        return []
    return [int(x) for x in out.split() if x.strip().isdigit()]


def _descendants(pid: int) -> list[int]:
    out: list[int] = []
    stack = list(_children(pid))
    while stack:
        c = stack.pop()
        out.append(c)
        stack.extend(_children(c))
    return out


def _signal_pid(pid: int, sig: signal.Signals, actions: list[dict], pattern: str, how: str) -> None:
    try:
        os.kill(pid, sig)
        actions.append({"action": how, "pid": pid, "pattern": pattern})
    except ProcessLookupError:
        pass


def stop_data_collection() -> list[dict]:
    """SIGTERM then SIGKILL matching roots and their descendant trees (vLLM kids)."""
    actions: list[dict] = []
    roots: list[tuple[int, str]] = []
    for pattern in KILL_PATTERNS:
        for pid in _pgrep(pattern):
            roots.append((pid, pattern))

    # Children first on TERM, then roots.
    for pid, pattern in roots:
        for child in reversed(_descendants(pid)):
            _signal_pid(child, signal.SIGTERM, actions, pattern, "sigterm_child")
        _signal_pid(pid, signal.SIGTERM, actions, pattern, "sigterm")

    time.sleep(5)

    for pid, pattern in roots:
        still = [c for c in _descendants(pid) if alive(c)]
        if alive(pid):
            still.append(pid)
        for victim in reversed(still):
            _signal_pid(victim, signal.SIGKILL, actions, pattern, "sigkill")
    return actions


def snapshot_gpu_memory() -> dict[int, int]:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    used: dict[int, int] = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            continue
        used[int(parts[0])] = int(float(parts[1]))
    return used


def compute_target_mib() -> dict[int, int]:
    """Return fixed per-GPU hold targets (never inflate from pre-kill used)."""
    return {g: int(DEFAULT_MIB[g]) for g in HOLD_GPUS}


def wait_for_memory_drop(before: dict[int, int], timeout_s: float = 60.0) -> dict[int, int]:
    """Poll until our GPUs shed most of the pre-kill footprint or timeout."""
    deadline = time.time() + timeout_s
    last = snapshot_gpu_memory()
    while time.time() < deadline:
        last = snapshot_gpu_memory()
        dropped = all(last.get(g, 0) <= max(2048, before.get(g, 0) // 3) for g in HOLD_GPUS)
        if dropped:
            return last
        time.sleep(2)
    return last


def start_gpu_holders(mib_by_gpu: dict[int, int]) -> int:
    HOLD_PID.parent.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in HOLD_GPUS)
    local_mib = {str(i): int(mib_by_gpu[g]) for i, g in enumerate(HOLD_GPUS)}
    env["CSD_GPU_HOLD_MIB_JSON"] = json.dumps(local_mib)
    ready = REPO / ".context" / "full-baseline-corrected-20260805-gpu-hold.ready"
    if ready.exists():
        ready.unlink()
    env["CSD_GPU_HOLD_READY"] = str(ready)
    log = open(LOG, "a")
    code = r"""
import json, os, signal, sys, time
from pathlib import Path
import torch

def _hold():
    mib = json.loads(os.environ["CSD_GPU_HOLD_MIB_JSON"])
    ready = Path(os.environ["CSD_GPU_HOLD_READY"])
    tensors = []
    try:
        for local, target in sorted(mib.items(), key=lambda kv: int(kv[0])):
            device = torch.device(f"cuda:{local}")
            n = max(1, int(int(target) * 1024 * 1024 / 2))
            tensors.append(torch.empty(n, dtype=torch.float16, device=device))
            torch.cuda.synchronize(device)
            print(f"[gpu-hold] local={local} target_mib={target} allocated", flush=True)
        ready.write_text("ok\n")
    except Exception as exc:
        ready.write_text(f"fail:{exc}\n")
        print(f"[gpu-hold] FAIL {exc}", flush=True)
        raise
    signal.pause()

_hold()
"""
    proc = subprocess.Popen(
        [sys.executable, "-c", code],
        cwd=str(REPO),
        stdout=log,
        stderr=log,
        start_new_session=True,
        env=env,
    )
    HOLD_PID.write_text(str(proc.pid) + "\n")
    return proc.pid


def verify_hold(hold_pid: int, target_mib: dict[int, int], timeout_s: float = 90.0) -> dict:
    ready = REPO / ".context" / "full-baseline-corrected-20260805-gpu-hold.ready"
    deadline = time.time() + timeout_s
    ready_text = ""
    while time.time() < deadline:
        if not alive(hold_pid):
            return {
                "ok": False,
                "reason": "hold_pid_dead",
                "hold_pid": hold_pid,
                "ready": ready.read_text().strip() if ready.exists() else "",
                "after_mib": snapshot_gpu_memory(),
            }
        if ready.exists():
            ready_text = ready.read_text().strip()
            if ready_text.startswith("fail:"):
                return {
                    "ok": False,
                    "reason": "holder_reported_fail",
                    "hold_pid": hold_pid,
                    "ready": ready_text,
                    "after_mib": snapshot_gpu_memory(),
                }
            if ready_text == "ok":
                break
        time.sleep(1)
    else:
        return {
            "ok": False,
            "reason": "ready_timeout",
            "hold_pid": hold_pid,
            "ready": ready_text,
            "after_mib": snapshot_gpu_memory(),
        }

    after = snapshot_gpu_memory()
    short = {
        g: after.get(g, 0)
        for g in HOLD_GPUS
        if after.get(g, 0) < int(target_mib[g] * MIN_HOLD_FRACTION)
    }
    if short:
        return {
            "ok": False,
            "reason": "after_mib_below_target",
            "hold_pid": hold_pid,
            "short_gpus": short,
            "target_mib": target_mib,
            "after_mib": after,
        }
    if not alive(hold_pid):
        return {
            "ok": False,
            "reason": "hold_pid_dead_after_alloc",
            "hold_pid": hold_pid,
            "after_mib": after,
        }
    return {"ok": True, "hold_pid": hold_pid, "after_mib": after, "target_mib": target_mib}


def write_pause_flag(**payload) -> None:
    body = {
        "paused_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "resume_requires": "human",
        "note": "Do not restart cold controller or authoring until this flag is removed.",
        **payload,
    }
    PAUSE_FLAG.write_text(json.dumps(body, indent=2) + "\n")


def check_hold_status() -> dict:
    """For post-pause babysit ticks: is the holder still alive and GPUs occupied?"""
    status: dict = {"pause_flag": PAUSE_FLAG.exists(), "hold_ok": False}
    if not PAUSE_FLAG.exists():
        status["reason"] = "no_pause_flag"
        return status
    flag = json.loads(PAUSE_FLAG.read_text())
    status["flag"] = {
        "reason": flag.get("reason"),
        "hold_pid": flag.get("hold_pid"),
        "hold_verified": flag.get("hold_verified"),
    }
    pid = int(flag.get("hold_pid") or 0)
    if HOLD_PID.exists():
        try:
            pid = int(HOLD_PID.read_text().strip() or pid)
        except ValueError:
            pass
    status["hold_pid"] = pid
    status["hold_alive"] = alive(pid)
    status["after_mib"] = snapshot_gpu_memory()
    target = {int(k): int(v) for k, v in (flag.get("target_mib") or {}).items()}
    if not target:
        target = compute_target_mib()
    short = {
        g: status["after_mib"].get(g, 0)
        for g in HOLD_GPUS
        if status["after_mib"].get(g, 0) < int(target.get(g, DEFAULT_MIB[g]) * MIN_HOLD_FRACTION)
    }
    status["short_gpus"] = short
    status["hold_ok"] = bool(status["hold_alive"] and not short and flag.get("hold_verified") is not False)
    if not status["hold_alive"]:
        status["reason"] = "hold_pid_dead"
    elif short:
        status["reason"] = "memory_below_target"
    else:
        status["reason"] = "ok"
    return status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reason", default="claude_usage_limit")
    parser.add_argument("--evidence", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check-hold", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if args.self_check:
        t = compute_target_mib()
        assert t == DEFAULT_MIB
        assert MIN_HOLD_FRACTION < 1.0
        print(json.dumps({"self_check": True, "target_mib": t}, indent=2))
        return 0

    if args.check_hold:
        status = check_hold_status()
        print(json.dumps(status, indent=2))
        return 0 if status.get("hold_ok") or not status.get("pause_flag") else 2

    before = snapshot_gpu_memory()
    mib = compute_target_mib()
    logging.info("pause begin reason=%s before_mib=%s target_mib=%s", args.reason, before, mib)

    if args.dry_run:
        print(json.dumps({"dry_run": True, "target_mib": mib, "before": before}, indent=2))
        return 0

    stop_actions = stop_data_collection()
    mid = wait_for_memory_drop(before)
    logging.info("post-kill mib=%s", mid)
    hold_pid = start_gpu_holders(mib)
    verification = verify_hold(hold_pid, mib)
    write_pause_flag(
        reason=args.reason,
        evidence=args.evidence,
        gpus_held=list(HOLD_GPUS),
        hold_pid=hold_pid,
        target_mib={str(k): v for k, v in mib.items()},
        stop_actions=stop_actions,
        hold_verified=bool(verification.get("ok")),
        verification=verification,
        before_mib={str(k): v for k, v in before.items()},
        mid_mib={str(k): v for k, v in mid.items()},
    )
    if not verification.get("ok"):
        logging.error("pause flag written but hold FAILED: %s", verification)
        print(json.dumps({"ok": False, "hold_pid": hold_pid, "verification": verification, "flag": str(PAUSE_FLAG)}, indent=2))
        return 1

    logging.info("pause armed hold_pid=%s verification=%s", hold_pid, verification)
    print(json.dumps({"ok": True, "hold_pid": hold_pid, "verification": verification, "flag": str(PAUSE_FLAG)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
