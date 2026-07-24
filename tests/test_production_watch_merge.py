"""Pre-merge snapshot of live dirty state, including untracked collisions."""

from __future__ import annotations

import subprocess
from pathlib import Path

from scripts.runtime.zero_acc_babysitter.production_hooks import (
    commit_live_dirty_state,
)


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    )


def _init_repo(root: Path) -> None:
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test")


def test_untracked_file_colliding_with_incoming_ref_is_committed(tmp_path: Path):
    """Regression: smiles-acrylates-qwen25-1p5b:7:telemetry:1784876387.

    Live carried an untracked file that the repair branch also added; git
    aborted the merge with "untracked working tree files would be
    overwritten". The pre-merge snapshot must stage such files so the merge
    proceeds, while leaving unrelated untracked files uncommitted.
    """
    repo = tmp_path / "live"
    repo.mkdir()
    _init_repo(repo)
    (repo / "base.txt").write_text("base\n")
    _git(repo, "add", "base.txt")
    _git(repo, "commit", "-m", "base")

    _git(repo, "checkout", "-b", "repair")
    (repo / "new_module.py").write_text("VALUE = 1\n")
    _git(repo, "add", "new_module.py")
    _git(repo, "commit", "-m", "repair adds module")
    _git(repo, "checkout", "main")

    # Live deploy state: same path untracked, plus an unrelated untracked log.
    (repo / "new_module.py").write_text("VALUE = 1\n")
    (repo / "scratch.log").write_text("noise\n")

    commit_live_dirty_state(repo, "test-incident", incoming_ref="repair")

    tracked = _git(repo, "ls-files").stdout.splitlines()
    assert "new_module.py" in tracked
    assert "scratch.log" not in tracked

    merged = subprocess.run(
        ["git", "merge", "--no-edit", "-X", "theirs", "repair"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert merged.returncode == 0, merged.stderr


def test_tracked_dirty_state_still_committed_without_incoming_ref(tmp_path: Path):
    repo = tmp_path / "live"
    repo.mkdir()
    _init_repo(repo)
    (repo / "base.txt").write_text("base\n")
    _git(repo, "add", "base.txt")
    _git(repo, "commit", "-m", "base")

    (repo / "base.txt").write_text("dirty\n")
    commit_live_dirty_state(repo, "test-incident")

    status = _git(repo, "status", "--porcelain").stdout.strip()
    assert status == ""


def test_clean_tree_is_a_noop(tmp_path: Path):
    repo = tmp_path / "live"
    repo.mkdir()
    _init_repo(repo)
    (repo / "base.txt").write_text("base\n")
    _git(repo, "add", "base.txt")
    _git(repo, "commit", "-m", "base")
    head = _git(repo, "rev-parse", "HEAD").stdout.strip()

    commit_live_dirty_state(repo, "test-incident", incoming_ref="HEAD")

    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == head
