# Babysitter repair via sibling git worktree

**Date:** 2026-07-24  
**What for:** Unblock Cursor CLI auto-repair without `git checkout` on the live cold-queue tree.  
**Branch / worktree:** `codex/babysitter-repair-worktree` at `worktrees/babysitter-repair-worktree`

## Result

- New helper: `scripts/runtime/zero_acc_babysitter/repair_worktree.py`
  - Default path: `<repo-parent>/<repo-name>-babysitter-repair` (e.g. `~/csd-generation-babysitter-repair`)
  - `ensure_repair_worktree` runs `git worktree add --detach` from live HEAD into that path
- Production watch (`production_watch.py`) now defaults to **auto-repair on** in that worktree:
  - Marker: `WAKE_AUTO_REPAIR_VIA_WORKTREE`
  - Runs `CursorCliClient.debug_fix` (checkout/commit/push/PR) only under the repair worktree
  - Does **not** merge into the live queue tree
- Observe-only: `--no-auto-repair`
- Custom path: `--repair-worktree /path`

## Tests

```bash
cd worktrees/babysitter-repair-worktree
/usr/local/bin/python3 -m pytest tests/test_babysitter_repair_worktree.py tests/test_cursor_cli_client.py -q
# 12 passed
```

Key falsifier: `test_debug_fix_checkout_does_not_move_live_branch` — live stays on `main` while repair worktree moves to `babysitter-fix/...`.

## Focal rollout (not done in this change)

1. Deploy this branch’s babysitter package to focal (or path-checkout the touched files).
2. Create worktree once if missing:  
   `git -C ~/csd-generation worktree add --detach ~/csd-generation-babysitter-repair HEAD`
3. Restart watcher **without** `--no-auto-repair` (new default).
4. Confirm `WATCHER_START` contains `auto_repair=worktree` and SMILES wakes show `WAKE_AUTO_REPAIR_VIA_WORKTREE` instead of `live_tree_unsafe_for_cli_checkout`.

## Context

Prior launch disarmed repair because `debug_fix` did `git checkout -B` in `workspace=live`. Sibling worktree is the git fix.
