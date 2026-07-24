# 2026-07-24 — babysitter merge deadlock unblock (spider-qwen25-1p5b:2:telemetry:1784857356)

Smoke passed on the PR tip at 05:22 UTC (acc 20%, rc=0) but `merge_and_pull`
failed: git refuses to merge over uncommitted live deploy state (any dirty
path the merge updates blocks it, even byte-identical content — verified on
git 2.34 with a synthetic repo).

## What this branch commit does

- Adopts live's uncommitted working copies verbatim (so the merge is
  conflict-free once live state is committed): `cloud.py`,
  `production_watch.py`, `repair_worktree.py`, `__main__.py` (Claude Code CLI
  migration), `run_cold_synthesis_queue.py` +
  `tests/runtime/test_cold_synthesis_queue.py` (gpu-mem-util refactor),
  babysitter `AGENTS.md`.
- Adds `_resolve_vllm_gpu_memory_utilization` to `synthesis/run_synthesis.py`
  (honors `CSD_VLLM_GPU_MEMORY_UTILIZATION` set by the queue scheduler; live's
  adopted test specified it but the function did not exist yet).
- `production_hooks.merge_pr_and_pull`: auto-commits tracked dirty live state
  before merging (never stash/discard) and merges `-X theirs`; aborts cleanly
  on failure.

## Intentionally NOT adopted from live

- `synthesis/scripts/reevaluate_compiled_csd.py`: live's dirty copy is a
  regression vs the snapshot base (drops `build_reevaluation_provenance` and
  the strict split-name flags that base tests require). Branch version wins
  via `-X theirs`. Live's `CSD_PARITY_SEED` block survives only in the
  pre-merge snapshot commit — re-land it separately if still wanted.
- Branch's earlier per-job `--vllm-gpu-memory-utilization` queue passing:
  superseded by live's env-var refactor.

## Retained branch fixes

- `smoke.py` `pick_smoke_gpu` / `SMOKE_GPU_MEM_UTIL=0.3` (live's copy is a
  strict ancestor, so `-X theirs` keeps the fix).

## Operator unblock

In the live checkout on `synthesis-snapshot-20260622`:
`git commit -a -m "snapshot live deploy state before babysitter merge"`,
then the watcher's next resume merges and recovery proceeds. No restart needed.
