"""Repair-agent hooks: Claude Code CLI (local headless) or local fake.

Callers: orchestrator BabysitterHooks.run_cloud_debug; focal mock suite.
API: RepairAgentClient protocol; NullCloudClient for local sim;
      ClaudeCodeCliClient for headless `claude -p` on a git checkout.
Docs (checked 2026-07-24): https://code.claude.com/docs/en/headless
  https://code.claude.com/docs/en/cli-reference
  https://code.claude.com/docs/en/model-config (claude-fable-5)
User instruction: babysitter must use Claude Code CLI + Fable 5 (not Cursor).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

from scripts.runtime.zero_acc_babysitter.persistence import IncidentRecord

LogEmit = Callable[[str, str, str], None]

# Anthropic / Claude Code model id for Fable 5 (docs 2026-07-24).
DEFAULT_CLAUDE_CODE_MODEL = "claude-fable-5"


class RepairAgentClient(Protocol):
    def debug_fix(self, incident: IncidentRecord) -> str | None:
        """Run a repair agent; return PR URL or branch:name if any."""


# Backward-compatible alias used by existing local sim / docs.
CloudClient = RepairAgentClient


class NullCloudClient:
    """Local-sim stub — never launches a real agent."""

    def debug_fix(self, incident: IncidentRecord) -> str | None:
        return f"https://example.test/pr/{incident.incident_id}"


NullRepairClient = NullCloudClient


def _safe_branch_slug(incident_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", incident_id).strip("-")
    return slug[:120] or "incident"


def _oauth_preferred_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """Env for Claude Code Max/OAuth: strip API-key vars so subscription wins.

    Docs (https://code.claude.com/docs/en/authentication): when
    ``ANTHROPIC_API_KEY`` / ``ANTHROPIC_AUTH_TOKEN`` are set they outrank
    subscription OAuth from ``claude auth login`` / ``/login``. Unset them so
    Max OAuth credentials are used. Keep ``CLAUDE_CODE_OAUTH_TOKEN`` if set
    (setup-token path).
    """
    env = dict(base if base is not None else os.environ)
    for key in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        env.pop(key, None)
    return env


def probe_claude_code_cli(*, agent_bin: str | None = None) -> tuple[bool, str]:
    """Return (ready_for_autonomous_repair, note).

    Ready means: ``claude`` binary exists AND ``claude auth status`` reports
    loggedIn under an OAuth-preferred env (API key stripped). Babysitter repair
    uses Claude Max subscription OAuth, not API-key credits.
    """
    bin_name = agent_bin or os.environ.get("CLAUDE_CODE_BIN", "claude")
    resolved = None
    if os.sep in bin_name or bin_name.startswith("."):
        if Path(bin_name).is_file():
            resolved = str(Path(bin_name).resolve())
    else:
        resolved = shutil.which(bin_name)
    if not resolved:
        return False, f"claude binary missing ({bin_name!r} not on PATH / not a file)"

    version = "unknown"
    try:
        ver = subprocess.run(
            [resolved, "--version"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        raw = (ver.stdout or ver.stderr or "").strip()
        version = raw.splitlines()[0] if raw else "unknown"
    except Exception as exc:  # noqa: BLE001
        return False, f"claude --version failed: {type(exc).__name__}: {exc}"

    oauth_env = _oauth_preferred_env()
    try:
        status = subprocess.run(
            [resolved, "auth", "status"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            env=oauth_env,
        )
    except Exception as exc:  # noqa: BLE001
        return False, (
            f"claude={resolved} version={version}; auth status failed: "
            f"{type(exc).__name__}: {exc}; run `claude auth login --claudeai`"
        )

    text = ((status.stdout or "") + (status.stderr or "")).strip()
    logged_in = False
    try:
        payload = json.loads(text)
        logged_in = bool(payload.get("loggedIn"))
    except json.JSONDecodeError:
        logged_in = (
            status.returncode == 0
            and "logged" in text.lower()
            and "not logged" not in text.lower()
        )

    if logged_in:
        return True, f"claude={resolved} version={version}; oauth_auth_ok={text[:200]!r}"
    return False, (
        f"claude={resolved} version={version}; not authenticated for Max OAuth "
        f"(status={text[:200]!r}). Run `claude auth login --claudeai` "
        f"(API key is intentionally ignored for babysitter)."
    )


@dataclass
class ClaudeCodeCliClient:
    """Headless Claude Code CLI repair against a local git workspace.

    Docs-backed flags: ``claude -p/--print --model claude-fable-5 --effort medium``
    with ``--dangerously-skip-permissions`` for unattended edits.
    Uses ``cwd=workspace`` (Claude has no ``--workspace`` flag).

    Auth: prefer Claude Max subscription OAuth (``claude auth login --claudeai``).
    Subprocess env strips ``ANTHROPIC_API_KEY`` / ``ANTHROPIC_AUTH_TOKEN`` so
    API-key credits do not outrank Max OAuth (docs auth precedence). Default
    is not ``--bare`` (bare is API-key oriented and skips OAuth token paths).

    Production callers must set ``workspace`` to a sibling repair worktree,
    not the live cold-queue checkout.
    """

    workspace: Path
    agent_bin: str = field(default_factory=lambda: os.environ.get("CLAUDE_CODE_BIN", "claude"))
    model: str | None = field(
        default_factory=lambda: os.environ.get("CLAUDE_CODE_MODEL")
        or os.environ.get("ANTHROPIC_MODEL")
        or DEFAULT_CLAUDE_CODE_MODEL
    )
    effort: str = field(
        default_factory=lambda: os.environ.get("CLAUDE_CODE_EFFORT", "medium")
    )
    timeout_sec: int = 1800
    create_pr: bool = True
    push: bool = True
    remote: str = "origin"
    base_ref: str = "HEAD"
    pr_base: str = field(
        default_factory=lambda: os.environ.get(
            "BABYSITTER_PR_BASE", "synthesis-snapshot-20260622"
        )
    )
    permission_mode: str = field(
        default_factory=lambda: os.environ.get(
            "CLAUDE_CODE_PERMISSION_MODE", "bypassPermissions"
        )
    )
    # None => False (OAuth Max default). Set True only for explicit API-key runs.
    use_bare: bool | None = None
    prefer_oauth: bool = True  # strip API key env for claude subprocess
    log_emit: LogEmit | None = None
    prompt_extra: str = ""

    def _emit(self, incident: IncidentRecord, marker: str, detail: str = "") -> None:
        if self.log_emit is not None:
            self.log_emit(incident.cell_id, marker, detail)

    def _build_prompt(self, incident: IncidentRecord) -> str:
        parts = [
            "You are the zero-acc babysitter repair agent running in Claude Code CLI headless mode.",
            "Debug and fix the failure for this incident. Prefer minimal, targeted edits.",
            f"incident_id={incident.incident_id}",
            f"cell_id={incident.cell_id}",
            f"attempt_index={incident.attempt_index}",
            f"path_kind={incident.path_kind}",
            f"broken_sha={incident.broken_sha or 'unknown'}",
            "After edits, leave the working tree ready to commit (do not push).",
        ]
        if self.prompt_extra.strip():
            parts.append(self.prompt_extra.strip())
        return "\n".join(parts)

    def _run_git(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=self.workspace,
            capture_output=True,
            text=True,
            check=False,
        )

    def _ensure_branch(self, incident: IncidentRecord) -> str:
        branch = f"babysitter-fix/{_safe_branch_slug(incident.incident_id)}"
        current = self._run_git("rev-parse", "--abbrev-ref", "HEAD")
        if current.returncode == 0 and (current.stdout or "").strip() == branch:
            return branch
        checkout = self._run_git("checkout", "-B", branch, self.base_ref)
        if checkout.returncode != 0:
            raise RuntimeError(
                f"git checkout -B {branch} failed: {(checkout.stderr or checkout.stdout)!r}"
            )
        return branch

    def _want_bare(self) -> bool:
        if self.use_bare is not None:
            return self.use_bare
        # OAuth Max is default: do not auto-enable --bare from ANTHROPIC_API_KEY.
        return False

    def _claude_env(self) -> dict[str, str]:
        if self.prefer_oauth:
            return _oauth_preferred_env()
        return dict(os.environ)

    def _agent_cmd(self, prompt: str) -> list[str]:
        cmd = [self.agent_bin]
        if self._want_bare():
            cmd.append("--bare")
        cmd.extend(
            [
                "-p",
                "--output-format",
                "text",
                "--no-session-persistence",
                "--dangerously-skip-permissions",
            ]
        )
        # permission-mode is redundant with dangerously-skip but kept configurable
        if self.permission_mode and self.permission_mode != "bypassPermissions":
            cmd.extend(["--permission-mode", self.permission_mode])
        if self.model:
            cmd.extend(["--model", self.model])
        if self.effort:
            cmd.extend(["--effort", self.effort])
        cmd.append(prompt)
        return cmd

    def _commit_if_dirty(self, incident: IncidentRecord) -> bool:
        status = self._run_git("status", "--porcelain")
        if status.returncode != 0:
            raise RuntimeError(f"git status failed: {(status.stderr or status.stdout)!r}")
        if not (status.stdout or "").strip():
            return False
        add = self._run_git("add", "-A")
        if add.returncode != 0:
            raise RuntimeError(f"git add failed: {(add.stderr or add.stdout)!r}")
        msg = f"babysitter Claude Code repair {incident.incident_id}"
        commit = self._run_git("commit", "-m", msg)
        if commit.returncode != 0:
            raise RuntimeError(f"git commit failed: {(commit.stderr or commit.stdout)!r}")
        return True

    def _push_and_pr(self, branch: str, incident: IncidentRecord) -> str | None:
        if self.push:
            push = self._run_git("push", "-u", self.remote, branch)
            if push.returncode != 0:
                self._emit(
                    incident,
                    "CLI_AGENT_PUSH_FAIL",
                    f"branch={branch} err={(push.stderr or push.stdout or '')[:300]!r}",
                )
                return f"branch:{branch}"
        if not self.create_pr:
            return f"branch:{branch}"
        title = f"babysitter fix: {incident.cell_id} ({incident.path_kind})"
        body = (
            f"Autonomous Claude Code CLI repair for incident `{incident.incident_id}`.\n\n"
            f"- cell: `{incident.cell_id}`\n"
            f"- path_kind: `{incident.path_kind}`\n"
            f"- broken_sha: `{incident.broken_sha}`\n"
            f"- model: `{self.model}`\n"
        )
        pr = subprocess.run(
            [
                "gh",
                "pr",
                "create",
                "--base",
                self.pr_base,
                "--title",
                title,
                "--body",
                body,
                "--head",
                branch,
            ],
            cwd=self.workspace,
            capture_output=True,
            text=True,
            check=False,
        )
        out = (pr.stdout or "").strip()
        if pr.returncode == 0 and out:
            self._emit(incident, "CLI_AGENT_PR", f"url={out.splitlines()[-1]}")
            return out.splitlines()[-1]
        self._emit(
            incident,
            "CLI_AGENT_PR_FAIL",
            f"err={(pr.stderr or pr.stdout or '')[:300]!r}",
        )
        return f"branch:{branch}"

    def debug_fix(self, incident: IncidentRecord) -> str | None:
        workspace = self.workspace.resolve()
        if not workspace.is_dir():
            raise RuntimeError(f"workspace missing: {workspace}")

        branch = self._ensure_branch(incident)
        prompt = self._build_prompt(incident)
        cmd = self._agent_cmd(prompt)
        self._emit(
            incident,
            "CLI_AGENT_START",
            f"branch={branch} bin={self.agent_bin} model={self.model} "
            f"bare={self._want_bare()} attempt={incident.cloud_attempt_count}",
        )
        self._emit(
            incident,
            "CLOUD_AGENT_START",
            f"via=claude_code branch={branch} model={self.model}",
        )

        try:
            proc = subprocess.run(
                cmd,
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
                check=False,
                env=self._claude_env(),
            )
        except subprocess.TimeoutExpired as exc:
            self._emit(incident, "CLI_AGENT_FAIL", f"timeout={self.timeout_sec}")
            self._emit(incident, "CLOUD_AGENT_FAIL", "timeout")
            raise RuntimeError(f"CLI_AGENT_FAIL: timeout after {self.timeout_sec}s") from exc

        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "")[:500]
            self._emit(incident, "CLI_AGENT_FAIL", f"rc={proc.returncode} err={detail!r}")
            self._emit(incident, "CLOUD_AGENT_FAIL", f"rc={proc.returncode}")
            raise RuntimeError(
                f"CLI_AGENT_FAIL: claude exited {proc.returncode}: {detail!r}"
            )

        dirty = self._commit_if_dirty(incident)
        self._emit(
            incident,
            "CLI_AGENT_DONE",
            f"branch={branch} committed={dirty} rc=0",
        )
        self._emit(incident, "CLOUD_AGENT_DONE", f"branch={branch}")

        if not dirty and not self.push:
            return f"branch:{branch}"
        return self._push_and_pr(branch, incident)


# Removed Cursor path — keep name only as explicit error if old imports linger.
def probe_cursor_cli(*_a, **_k):  # noqa: ANN001
    raise RuntimeError(
        "Cursor CLI repair is removed; use probe_claude_code_cli / ClaudeCodeCliClient"
    )


class CursorCliClient:  # pragma: no cover — hard fail on legacy use
    def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError(
            "CursorCliClient removed; use ClaudeCodeCliClient (model claude-fable-5)"
        )
