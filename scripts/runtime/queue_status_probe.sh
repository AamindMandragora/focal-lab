#!/usr/bin/env bash
# Compact status probe for the seven-row recovery queue. Prints a small block;
# the Mac-side watcher dedupes and only surfaces changes.
set -u
cd /home/aadivyar/csd-generation || { echo "PROBE_ERROR cd failed"; exit 1; }
Q=$(systemctl --user is-active csd-claude-recovery-queue.service 2>&1)
M=$(systemctl --user is-active csd-codex-incident-monitor.service 2>&1)
[ "$Q" = "active" ] || echo "SVC_ERROR queue=$Q"
[ "$M" = "active" ] || echo "SVC_ERROR incident-monitor=$M"
/apps/conda/aadivyar/envs/csd/bin/python - <<'PY'
import json, re, subprocess, sys
from pathlib import Path
sys.path.insert(0, "/home/aadivyar/csd-generation")
from scripts.runtime import supervise_warm_task_recovery as sup

repo = Path("/home/aadivyar/csd-generation")
jobs = sup.load_manifest(repo / "saved-results/2026-07-15-claude-helper-recovery-manifest.json")
jobs = sup.apply_conditional_extensions(repo, jobs)
terminal = {sup.COMPLETE_SUCCESS, sup.COMPLETE_FAILURE}
phases = {}
for job in jobs:
    cell = str(job["cell_id"])
    phase = sup.job_phase(repo, job)
    phases[cell] = phase
    out = str(job["output_name"])
    anchor = ""
    log = repo / f"logs/paid_synth_{out}.log"
    if log.is_file():
        with log.open("rb") as f:
            f.seek(max(0, log.stat().st_size - 5_000_000))
            tail = f.read().decode("utf-8", "replace")
        # Search the whole log backwards for the newest anchor; tac stops at
        # the first grep match, so this stays cheap even on multi-GB logs.
        found = subprocess.run(
            f"tac '{log}' | grep -m1 -o 'anchor for next refinement: attempt [0-9]* (acc=[0-9.]*%, syn=[0-9.]*%)'",
            shell=True, capture_output=True, text=True,
        ).stdout.strip()
        hit = re.search(r"attempt (\d+) \(acc=([\d.]+)%, syn=([\d.]+)%\)", found)
        if hit:
            a, acc, syn = hit.groups()
            anchor = f" att={a} acc={acc}% syn={syn}%"
        fails = re.findall(r"\[claude\] failure exit_status=\d+ category=[\w-]+", tail)
        if fails:
            anchor += f" author_fails={len(fails)}"
        last_lines = tail.splitlines()[-200:]
        errs = [l for l in last_lines if re.search(r"Traceback|CUDA out of memory|NO_ACCEPTED_CSD|FATAL", l)]
        if errs:
            anchor += f" LOG_ERR:{errs[-1][:120]}"
    print(f"ROW {cell} {phase} cap={job.get('total_cap')}{anchor}")
if phases and all(p in terminal for p in phases.values()):
    print("ALL_TERMINAL " + " ".join(f"{c}={p}" for c, p in sorted(phases.items())))
inc = repo / ".context/codex_incident_monitor"
if inc.is_dir():
    dirs = sorted(d.name for d in inc.iterdir() if d.is_dir())
    print(f"INCIDENTS n={len(dirs)} last={dirs[-1] if dirs else 'none'}")
PY
