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
4. Claude Code CLI `debug_fix` in repair worktree (not live checkout)
5. Hardened smoke on PR tip — **no merge on FAIL**
6. Smoke PASS → `merge_and_pull` into `synthesis-snapshot-20260622` → live pull → `MERGED sha=...`
7. Recovery: harness/telemetry → rescore_all + siblings; helper → restart_from_K; memory → resume

Max 30 Cloud attempts per incident; mid-incident resume from JSON; ≤1 active
incident per cell; `acc >= 15` → no wake.

## Rules

- Never run `ClaudeCodeCliClient.debug_fix` with `workspace` set to the live cold-queue
  checkout. Use `repair_worktree.ensure_repair_worktree` (default
  `<repo-parent>/<repo-name>-babysitter-repair`).
- Repair PRs must target `--base synthesis-snapshot-20260622` (override via
  `BABYSITTER_PR_BASE`), never `master`/`main`.
- Observe-only: `--no-auto-repair` (no Cloud/smoke/merge).
- Default repair model: Claude Fable 5 (`claude-fable-5`) via Claude Code CLI.
- Local sim (`--local-sim-scenario`) never authorizes a real queue.

## Smoke gate (locked)

- SMILES harness: `unique_valid_count > 0` (or rate > 0). Acc=0 + UV=0 + uniq_tokens>0 = **FAIL**.
- GSM/Spider harness: Acc > 0%.
- Helper: micros empty. Telemetry: Accuracy line. Memory: no OOM.
- No measured metrics → refuse merge (unless `BABYSITTER_SMOKE_STUB=pass` dry-run).
- SMOKE_FAIL → no merge; Cloud retries up to 30.

## Sandbox

```bash
/usr/local/bin/python3 -m pytest \
  tests/test_zero_acc_babysitter_local_sim.py \
  tests/test_production_watch_merge.py \
  tests/test_babysitter_repair_worktree.py -q

PYTHONPATH=. /usr/local/bin/python3 -m scripts.runtime.zero_acc_babysitter \
  --local-sim-scenario tier_a_harness
```

## Bootstrap status — spider-qwen25-1p5b:2:telemetry:1784857356

RESOLVED 2026-07-24 ~06:00: the merge deadlock unblock has been **executed**
— live's dirty deploy state was committed on `synthesis-snapshot-20260622`
(`092fa159` snapshot + `2f320853` sync of smoke.py / production_hooks.py /
this file to the repair-branch tip), and the exact watcher merge command
(`git merge --no-edit origin/<repair-branch>`) was simulated in a detached
worktree: rc=0, zero conflicts. The watcher's next resume merges cleanly —
do NOT re-run the unblock or restart the watcher for this incident.

Lessons kept for future incidents:
- git refuses to merge over ANY dirty path it updates, even byte-identical
  ones — `merge_pr_and_pull` now auto-commits tracked dirty live state and
  merges `-X theirs` (live's side stays recoverable in the snapshot commit).
- git also aborts a merge over UNTRACKED live files the merge would write
  ("untracked working tree files would be overwritten", incident
  smiles-acrylates-qwen25-1p5b:7:telemetry:1784876387) — `commit_live_dirty_state`
  now takes `incoming_ref` and stages colliding untracked paths into the
  snapshot commit; unrelated untracked files stay uncommitted.
- Files that are add/add vs the merge base (e.g. this file,
  `production_hooks.py`) conflict whenever live and branch copies differ at
  all; when editing them on a repair branch, sync the same bytes to live and
  commit there so the plain merge stays trivial.
- See `saved-results/2026-07-24-babysitter-merge-deadlock-unblock.md` for
  fixes dropped from the merge result to re-land (none load-bearing).

## See also

- `production_watch.py`, `production_hooks.py`, `orchestrator.py`, `cloud.py`
- `planning/zero-acc-cloud-babysitter-plan.md`
- `saved-results/2026-07-24-babysitter-e2e-merge-wiring.md`
