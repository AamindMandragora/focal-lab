"""Unit tests for CursorCliClient (mocked agent binary; no real Cursor auth)."""

from __future__ import annotations

import stat
import subprocess
import textwrap
from pathlib import Path

import pytest

from scripts.runtime.zero_acc_babysitter.cloud import (
    DEFAULT_CURSOR_AGENT_MODEL,
    CursorCliClient,
    NullCloudClient,
    probe_cursor_cli,
)
from scripts.runtime.zero_acc_babysitter.persistence import IncidentRecord


def test_default_repair_model_is_cursor_grok_4_5_high() -> None:
    """Babysitter repairs must pin Cursor Grok 4.5 (verified via `agent --list-models`)."""
    assert DEFAULT_CURSOR_AGENT_MODEL == "cursor-grok-4.5-high"
    client = CursorCliClient(workspace=Path("."))
    assert client.model == "cursor-grok-4.5-high"
    cmd = client._agent_cmd("prompt")
    assert "--model" in cmd
    assert cmd[cmd.index("--model") + 1] == "cursor-grok-4.5-high"


def test_cursor_agent_model_env_overrides_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CURSOR_AGENT_MODEL", "composer-2.5")
    client = CursorCliClient(workspace=Path("."))
    assert client.model == "composer-2.5"
    cmd = client._agent_cmd("prompt")
    assert cmd[cmd.index("--model") + 1] == "composer-2.5"


def _incident(cell_id: str = "gsm-qwen25-1p5b") -> IncidentRecord:
    return IncidentRecord(
        incident_id=f"{cell_id}:1:harness:1700000000",
        cell_id=cell_id,
        attempt_index=1,
        path_kind="harness",
        trigger_unix_ts=1700000000,
        broken_sha="deadbeef",
    )


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


def _write_fake_agent(bin_dir: Path, *, fail: bool = False, touch: str = "FIXED.txt") -> Path:
    path = bin_dir / "fake-agent"
    # touch path is expanded by the fake agent after cd into workspace via --workspace arg parsing.
    body = textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        set -euo pipefail
        printf '%s\\n' "$@" > "$(dirname "$0")/last_args.txt"
        if [[ "$*" == *"--version"* ]]; then
          echo "2026.07.23-e383d2b-fake"
          exit 0
        fi
        if [[ "$*" == *"status"* ]]; then
          echo "Logged in as fake@example.com"
          exit 0
        fi
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
        {"exit 1" if fail else f'echo fixed > "$WORKSPACE/{Path(touch).name}"'}
        exit 0
        """
    )
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_null_cloud_client_returns_example_pr() -> None:
    url = NullCloudClient().debug_fix(_incident())
    assert url and url.startswith("https://example.test/pr/")


def test_cursor_cli_client_runs_agent_and_commits(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    agent = _write_fake_agent(tmp_path, touch="FIXED.txt")
    events: list[tuple[str, str, str]] = []

    client = CursorCliClient(
        workspace=repo,
        agent_bin=str(agent),
        create_pr=False,
        push=False,
        log_emit=lambda cell, marker, detail="": events.append((cell, marker, detail)),
    )
    url = client.debug_fix(_incident())
    assert url is not None
    assert url.startswith("branch:")
    assert (repo / "FIXED.txt").is_file()
    markers = [e[1] for e in events]
    assert "CLI_AGENT_START" in markers
    assert "CLI_AGENT_DONE" in markers
    args = (tmp_path / "last_args.txt").read_text(encoding="utf-8")
    assert "-p" in args or "--print" in args
    assert "--force" in args or "-f" in args
    assert "--trust" in args
    assert "--model" in args
    assert "cursor-grok-4.5-high" in args


def test_cursor_cli_client_logs_fail_on_nonzero(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    agent = _write_fake_agent(tmp_path, fail=True)
    events: list[tuple[str, str, str]] = []
    client = CursorCliClient(
        workspace=repo,
        agent_bin=str(agent),
        create_pr=False,
        push=False,
        log_emit=lambda cell, marker, detail="": events.append((cell, marker, detail)),
    )
    with pytest.raises(RuntimeError, match="CLI_AGENT_FAIL"):
        client.debug_fix(_incident())
    assert any(m == "CLI_AGENT_FAIL" for _, m, _ in events)


def test_probe_cursor_cli_reports_missing_binary(tmp_path: Path) -> None:
    ok, note = probe_cursor_cli(agent_bin=str(tmp_path / "no-such-agent"))
    assert ok is False
    assert "missing" in note.lower() or "not found" in note.lower()


def test_probe_cursor_cli_ok_with_fake_logged_in(tmp_path: Path) -> None:
    agent = _write_fake_agent(tmp_path)
    ok, note = probe_cursor_cli(agent_bin=str(agent))
    assert ok is True
