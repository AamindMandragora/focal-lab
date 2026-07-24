"""Git worktree used for Cursor CLI repair so live queue tree is never checked out.

Callers: production_watch.run_watch_loop / handle_watch_wake; tests.
API:
  default_repair_worktree_path(live_repo) -> sibling path
  ensure_repair_worktree(live_repo, worktree_path) -> resolved path
  live_head_sha(repo) -> HEAD sha string

User instruction: implement repair via separate git worktree (not live checkout).
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def default_repair_worktree_path(live_repo: Path) -> Path:
    live = live_repo.resolve()
    return live.parent / f"{live.name}-babysitter-repair"


def live_head_sha(repo: Path) -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"git rev-parse HEAD failed in {repo}: {(proc.stderr or proc.stdout)!r}"
        )
    return (proc.stdout or "").strip()


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )


def ensure_repair_worktree(live_repo: Path, worktree_path: Path) -> Path:
    """Create or reuse a linked worktree for repair; never checkout the live tree.

    If ``worktree_path`` already exists as a worktree/repo, leave its branch alone
    (repair checkouts happen later inside ClaudeCodeCliClient on that path only).
    """
    live = live_repo.resolve()
    repair = worktree_path.resolve()
    if repair == live:
        raise ValueError("repair worktree path must differ from live repo")

    git_meta = live / ".git"
    if not git_meta.exists():
        raise RuntimeError(f"live repo is not a git checkout: {live}")

    if repair.exists():
        probe = _git(repair, "rev-parse", "--is-inside-work-tree")
        if probe.returncode != 0 or (probe.stdout or "").strip() != "true":
            raise RuntimeError(
                f"repair path exists but is not a git worktree: {repair}"
            )
        return repair

    sha = live_head_sha(live)
    repair.parent.mkdir(parents=True, exist_ok=True)
    add = _git(live, "worktree", "add", "--detach", str(repair), sha)
    if add.returncode != 0:
        raise RuntimeError(
            f"git worktree add failed: {(add.stderr or add.stdout)!r}"
        )
    return repair
