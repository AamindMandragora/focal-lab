# AGENTS.md — `scripts/runtime/zero_acc_babysitter/`

## Scope

Zero-acc cold-queue babysitter: watch Acc/memory/TELEMETRY markers, run Cursor
CLI repair in a sibling worktree, hardened smoke, merge into
`synthesis-snapshot-20260622`, recover (rescore / restart-K / memory resume).

## Production path (locked)

`production_watch` → `Orchestrator` → `BabysitterHooks` from `production_hooks`:

1. Wake (MEMORY first, then Acc tier / TELEMETRY_FAIL)
2. Kill cell process group; record `broken_sha`
3. Tier A suite (twin + helper micros) / Tier B helpers / STRATEGY_MISS
4. Cursor CLI `debug_fix` in repair worktree (not live checkout)
5. Hardened smoke on PR tip — **no merge on FAIL**
6. Smoke PASS → `merge_and_pull` into `synthesis-snapshot-20260622` → live pull → `MERGED sha=...`
7. Recovery: harness/telemetry → rescore_all + siblings; helper → restart_from_K; memory → resume

Max 30 Cloud attempts per incident; mid-incident resume from JSON; ≤1 active
incident per cell; `acc >= 15` → no wake.

## Rules

- Never run `CursorCliClient.debug_fix` with `workspace` set to the live cold-queue
  checkout. Use `repair_worktree.ensure_repair_worktree` (default
  `<repo-parent>/<repo-name>-babysitter-repair`).
- Repair PRs must target `--base synthesis-snapshot-20260622` (override via
  `BABYSITTER_PR_BASE`), never `master`/`main`.
- Observe-only: `--no-auto-repair` (no Cloud/smoke/merge).
- Default repair model: Cursor Grok 4.5 (`cursor-grok-4.5-high`).
- Local sim (`--local-sim-scenario`) never authorizes a real queue.

## Smoke gate (locked)

- SMILES harness: `unique_valid_count > 0` (or rate > 0). Acc=0 + UV=0 + uniq_tokens>0 = **FAIL**.
- GSM/Spider harness: Acc > 0%.
- Helper: micros empty. Telemetry: Accuracy line. Memory: no OOM.
- No measured metrics → refuse merge (unless `BABYSITTER_SMOKE_STUB=pass` dry-run).
- SMOKE_FAIL → no merge; Cloud retries up to 30.

## Sandbox

```bash
python -m pytest \
  tests/test_zero_acc_babysitter_local_sim.py \
  tests/runtime/test_smoke_stale_metrics.py \
  tests/runtime/test_reevaluation_provenance.py -q

PYTHONPATH=. python -m scripts.runtime.zero_acc_babysitter \
  --local-sim-scenario tier_a_harness
```

## Bootstrap recovery (smoke-infra fix deadlock)

If the smoke *infrastructure itself* is what is broken (incident
spider-qwen25-1p5b:2:telemetry:1784857356), the loop deadlocks: repair commits
can only reach live via a smoke-gated merge, but the running watcher keeps
executing its cached pre-fix decide (it reuses stale `smoke_process_rc` from
`incident.extra` and never re-runs the smoke — instant same-second SMOKE_FAIL,
no new `smoke_*` out dir). No cloud attempt can fix a running process; operator
action is required:

1. Stop the watcher (it caches modules at first use; commits are invisible to it).
2. Either pull the repair branch's babysitter fixes into the live checkout, or
   delete the stale `smoke_report_path` / `smoke_process_rc` / `smoke_metrics`
   keys from `logs/zero_acc_babysitter/incidents/<incident>.json` so the
   resumed decide re-runs a real smoke (the PR-tip reevaluate defaults the
   split side to `train` for babysitter smoke report paths).
3. Restart the watcher; resume re-runs the smoke on the PR tip and the normal
   gate decides the merge. Never hand-write passing smoke metrics.

Caution for step 2: do NOT pull/copy the whole repair branch into the live
checkout — the live working tree can carry newer uncommitted deploy state (it
was ahead on the Cursor→Claude Code CLI migration in `cloud.py` /
`production_watch.py` / `__main__.py` on 2026-07-24). Diff each file first and
sync only the fix-carrying files whose live copy has no unique lines
(`smoke.py`, `production_hooks.py` for this incident). Restart may be scheduled
detached (`systemd-run --user --on-active=N systemctl --user restart
csd-zero-acc-babysitter.service`) since a direct restart kills the in-cgroup
repair agent (`KillMode=control-group`).

## See also

- `production_watch.py`, `production_hooks.py`, `orchestrator.py`, `cloud.py`
- `planning/zero-acc-cloud-babysitter-plan.md`
- `saved-results/2026-07-24-babysitter-e2e-merge-wiring.md`
