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

2026-07-24 ~05:30: smoke PASSED on the PR tip (acc 20%, rc=0,
`smoke_spider-qwen25-1p5b_20260724T045916Z`), but `merge_and_pull` failed:
git refuses to merge over uncommitted live deploy state (it refuses for any
dirty path the merge updates, even byte-identical ones). Unblock, in the
live checkout on `synthesis-snapshot-20260622`:

```bash
git commit -a -m "snapshot live deploy state before babysitter merge"
```

then let the watcher's next resume retry the merge (no restart needed). The
repair branch now carries live's uncommitted Claude-CLI migration and
gpu-scheduling edits verbatim, so the merge resolves cleanly; going forward
`merge_pr_and_pull` auto-commits tracked dirty live state and merges with
`-X theirs` (live's side stays recoverable in the snapshot commit). See
`saved-results/2026-07-24-babysitter-merge-deadlock-unblock.md` for fixes
dropped from the merge result that should be re-landed (none load-bearing).

## See also

- `production_watch.py`, `production_hooks.py`, `orchestrator.py`, `cloud.py`
- `planning/zero-acc-cloud-babysitter-plan.md`
- `saved-results/2026-07-24-babysitter-e2e-merge-wiring.md`
