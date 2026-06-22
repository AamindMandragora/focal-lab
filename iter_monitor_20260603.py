#!/usr/bin/env python3
"""Per-iteration monitor for the 4-stage iter30 chain (2026-06-03).
Polls each stage log; emits ONE stdout line per newly-completed attempt with
(attempt#, max, accuracy, syntax, pass/fail). State persisted across restarts.
"""
import json
import os
import re
import sys
import time

STATE_FILE = "/tmp/iter_monitor_state_iter30_20260603.json"
POLL_SECONDS = 60

STAGES = [
    ("GSM-1.5B-iter30",   "/tmp/ralph_1p5B_gsm_disjoint_iter30_20260603.log"),
    ("Spider-1.5B-iter30","/tmp/ralph_1p5B_spider_disjoint_iter30_20260603.log"),
    ("Spider-7B-iter30",  "/tmp/ralph_7B_spider_disjoint_iter30_20260603.log"),
    ("GSM-7B-iter30",     "/tmp/ralph_7B_gsm_disjoint_iter30_20260603.log"),
]

ATTEMPT_RE = re.compile(r"^Attempt (\d+)/(\d+)\s*$")
ACC_RE = re.compile(r"Accuracy:\s*([\d.]+)%")
SYN_RE = re.compile(r"Syntax:\s*([\d.]+)%")
THRESH_RE = re.compile(r"(✗ Evaluation below threshold|✓ Evaluation passed)")


def parse(path):
    """Return list of dicts for every completed attempt: {attempt, max, acc, syn, status}."""
    try:
        with open(path, "r", errors="replace") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return []
    out = []
    cur = None
    in_eval_block = False
    for line in lines:
        m = ATTEMPT_RE.match(line.strip())
        if m:
            if cur and cur.get("acc") is not None and cur.get("syn") is not None:
                out.append(cur)
            cur = {
                "attempt": int(m.group(1)),
                "max": int(m.group(2)),
                "acc": None,
                "syn": None,
                "status": None,
            }
            in_eval_block = False
            continue
        if cur is None:
            continue
        tm = THRESH_RE.search(line)
        if tm:
            cur["status"] = "pass" if "✓" in tm.group(0) else "fail"
            in_eval_block = True
            continue
        if in_eval_block:
            am = ACC_RE.search(line)
            if am and cur["acc"] is None:
                cur["acc"] = float(am.group(1))
            sm = SYN_RE.search(line)
            if sm and cur["syn"] is None:
                cur["syn"] = float(sm.group(1))
                in_eval_block = False  # both metrics captured
    if cur and cur.get("acc") is not None and cur.get("syn") is not None:
        out.append(cur)
    return out


def load_state():
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(s):
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(s, f)
    os.replace(tmp, STATE_FILE)


def history_str(iters, n=3):
    tail = iters[-n:]
    return ",".join(f"a{x['attempt']}=({x['acc']:.1f}/{x['syn']:.1f})" for x in tail)


def main():
    state = load_state()
    print(f"[MONITOR] iter30 per-iteration monitor started; state={STATE_FILE}", flush=True)
    while True:
        for name, path in STAGES:
            iters = parse(path)
            if not iters:
                continue
            last_emit = state.get(name, 0)
            new_ones = [it for it in iters if it["attempt"] > last_emit]
            for it in new_ones:
                hist = history_str(iters[: iters.index(it) + 1], n=4)
                print(
                    f"[ITER] {name} attempt={it['attempt']}/{it['max']} "
                    f"acc={it['acc']:.1f}% syn={it['syn']:.1f}% status={it['status']} "
                    f"history=[{hist}]",
                    flush=True,
                )
                state[name] = it["attempt"]
                save_state(state)
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
