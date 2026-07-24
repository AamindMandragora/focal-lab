"""Repair worktree isolates git checkout from the live cold-queue tree."""

from __future__ import annotations

import subprocess
import stat
import textwrap
from pathlib import Path

import pytest

from scripts.runtime.zero_acc_babysitter.cloud import CursorCliClient
from scripts.runtime.zero_acc_babysitter.persistence import IncidentRecord
from scripts.runtime.zero_acc_babysitter.repair_worktree import (
    default_repair_worktree_path,
    ensure_repair_worktree,
    live_head_sha,
)
from scripts.runtime.zero_acc_babysitter.production_watch import handle_watch_wake


def _init_git_repo(root: Path) -> None:
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    (root / "README.md").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=root, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    # Detached default branch names vary; force a stable branch name.
    subprocess.run(
        ["git", "branch", "-M", "main"],
        cwd=root,
        check=True,
        capture_output=True,
    )


def _git(cwd: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return (proc.stdout or "").strip()


def _write_fake_agent(bin_dir: Path) -> Path:
    path = bin_dir / "fake-agent"
    body = textwrap.dedent(
        """\
        #!/usr/bin/env bash
        set -euo pipefail
        WORKSPACE=""
        prev=""
        for arg in "$@"; do
          if [[ "$prev" == "--workspace" ]]; then
            WORKSPACE="$arg"
          fi
          prev="$arg"
        done
        if [[ -z "$WORKSPACE" ]]; then
          WORKSPACE="."
        fi
        echo fixed > "$WORKSPACE/FIXED.txt"
        exit 0
        """
    )
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _incident() -> IncidentRecord:
    return IncidentRecord(
        incident_id="smiles-acrylates-qwen25-1p5b:2:harness:1700000000",
        cell_id="smiles-acrylates-qwen25-1p5b",
        attempt_index=2,
        path_kind="harness",
        trigger_unix_ts=1700000000,
        broken_sha="deadbeef",
    )


def test_default_repair_worktree_path_is_sibling_not_live(tmp_path: Path) -> None:
    live = tmp_path / "csd-generation"
    live.mkdir()
    path = default_repair_worktree_path(live)
    assert path != live.resolve()
    assert path.name == "csd-generation-babysitter-repair"
    assert path.parent == live.resolve().parent


def test_ensure_repair_worktree_creates_sibling_and_keeps_live_branch(
    tmp_path: Path,
) -> None:
    live = tmp_path / "csd-generation"
    live.mkdir()
    _init_git_repo(live)
    repair = tmp_path / "csd-generation-babysitter-repair"

    out = ensure_repair_worktree(live, repair)
    assert out == repair.resolve()
    assert repair.is_dir()
    assert _git(live, "rev-parse", "--abbrev-ref", "HEAD") == "main"
    # Worktree is a linked checkout of the same commit.
    assert live_head_sha(live) == live_head_sha(repair)


def test_debug_fix_checkout_does_not_move_live_branch(tmp_path: Path) -> None:
    live = tmp_path / "csd-generation"
    live.mkdir()
    _init_git_repo(live)
    repair = tmp_path / "csd-generation-babysitter-repair"
    ensure_repair_worktree(live, repair)
    agent = _write_fake_agent(tmp_path)

    client = CursorCliClient(
        workspace=repair,
        agent_bin=str(agent),
        create_pr=False,
        push=False,
        base_ref=live_head_sha(live),
    )
    url = client.debug_fix(_incident())
    assert url is not None
    assert _git(live, "rev-parse", "--abbrev-ref", "HEAD") == "main"
    assert (live / "FIXED.txt").exists() is False
    assert (repair / "FIXED.txt").is_file()
    repair_branch = _git(repair, "rev-parse", "--abbrev-ref", "HEAD")
    assert repair_branch.startswith("babysitter-fix/")


def test_ensure_rejects_same_path_as_live(tmp_path: Path) -> None:
    live = tmp_path / "csd-generation"
    live.mkdir()
    _init_git_repo(live)
    with pytest.raises(ValueError, match="must differ"):
        ensure_repair_worktree(live, live)


def test_handle_watch_wake_auto_repair_uses_client_workspace(
    tmp_path: Path,
) -> None:
    """Observable: wake with auto_repair calls debug_fix; live branch unchanged."""
    live = tmp_path / "csd-generation"
    live.mkdir()
    _init_git_repo(live)
    repair = tmp_path / "csd-generation-babysitter-repair"
    ensure_repair_worktree(live, repair)

    called: list[str] = []

    class FakeClient:
        workspace = repair

        def debug_fix(self, incident: IncidentRecord) -> str | None:
            called.append(incident.incident_id)
            # Simulate the real client's branch switch only in repair tree.
            subprocess.run(
                ["git", "checkout", "-B", "babysitter-fix/fake", "HEAD"],
                cwd=repair,
                check=True,
                capture_output=True,
            )
            return "https://example.test/pr/1"

    events: list[tuple[str, str, str]] = []

    def emit(cell: str, marker: str, detail: str = "") -> None:
        events.append((cell, marker, detail))

    handle_watch_wake(
        live_repo=live,
        cell_id="smiles-acrylates-qwen25-1p5b",
        attempt_index=2,
        accuracy_pct=0.0,
        memory_ops=False,
        client=FakeClient(),  # type: ignore[arg-type]
        repair_worktree=repair,
        auto_repair=True,
        emit=emit,
    )
    assert called
    assert _git(live, "rev-parse", "--abbrev-ref", "HEAD") == "main"
    markers = [m for _, m, _ in events]
    assert "WAKE_AUTO_REPAIR_VIA_WORKTREE" in markers
    assert "WAKE_OBSERVED_NO_AUTO_REPAIR" not in markers
