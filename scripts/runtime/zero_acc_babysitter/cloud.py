"""Repair-agent hooks: Cursor CLI (local headless) or local fake.

Callers: orchestrator BabysitterHooks.run_cloud_debug; focal mock suite.
API: RepairAgentClient protocol; NullCloudClient for local sim;
      CursorCliClient for headless `agent -p --force` on a git checkout.
Docs (checked 2026-07-24): https://cursor.com/docs/cli/headless
  https://cursor.com/docs/cli/reference/parameters
  https://cursor.com/docs/cli/reference/authentication
User instruction: babysitter Cursor CLI instead of Cloud Agents API.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

from scripts.runtime.zero_acc_babysitter.persistence import IncidentRecord

LogEmit = Callable[[str, str, str], None]

# Verified 2026-07-24 via `agent --list-models` after login (CLI slug, not Task-tool-only).
# Docs: https://cursor.com/docs/cli/reference/parameters (`--model`)
DEFAULT_CURSOR_AGENT_MODEL = "cursor-grok-4.5-high"


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


def probe_cursor_cli(*, agent_bin: str | None = None) -> tuple[bool, str]:
    """Return (ready_for_autonomous_repair, note).

    Ready means: agent binary exists AND (CURSOR_API_KEY / CURSOR_AUTH_TOKEN
    set OR `agent status` reports logged in). Does not call Cloud Agents API.
    """
    bin_name = agent_bin or os.environ.get("CURSOR_AGENT_BIN", "agent")
    resolved = None
    if os.sep in bin_name or bin_name.startswith("."):
        if Path(bin_name).is_file():
            resolved = str(Path(bin_name).resolve())
    else:
        resolved = shutil.which(bin_name)
    if not resolved:
        return False, f"agent binary missing ({bin_name!r} not on PATH / not a file)"

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
        return False, f"agent --version failed: {type(exc).__name__}: {exc}"

    key = os.environ.get("CURSOR_API_KEY", "").strip()
    auth_token = os.environ.get("CURSOR_AUTH_TOKEN", "").strip()
    if key:
        return True, f"agent={resolved} version={version}; CURSOR_API_KEY set"
    if auth_token:
        return True, f"agent={resolved} version={version}; CURSOR_AUTH_TOKEN set"

    try:
        status = subprocess.run(
            [resolved, "status"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            env={**os.environ},
        )
    except Exception as exc:  # noqa: BLE001
        return False, (
            f"agent={resolved} version={version}; status failed: "
            f"{type(exc).__name__}: {exc}; set CURSOR_API_KEY / CURSOR_AUTH_TOKEN "
            "or run agent login"
        )

    text = ((status.stdout or "") + (status.stderr or "")).strip()
    lower = text.lower()
    if status.returncode == 0 and "not logged in" not in lower and text:
        return True, f"agent={resolved} version={version}; status_ok={text[:200]!r}"
    return False, (
        f"agent={resolved} version={version}; not authenticated "
        f"(status={text[:200]!r}). Set CURSOR_API_KEY / CURSOR_AUTH_TOKEN "
        "or run `agent login`. "
        "CLI does not use OPENAI_API_KEY/ANTHROPIC_API_KEY directly."
    )


@dataclass
class CursorCliClient:
    """Headless Cursor Agent CLI repair against a local git workspace.

    Docs-backed flags: ``agent -p/--print --force/--yolo --trust --workspace``
    plus ``--model`` (default Cursor Grok 4.5). Auth: ``CURSOR_API_KEY``,
    ``CURSOR_AUTH_TOKEN``, or prior ``agent login``.
    This is NOT the Cloud Agents API; it runs on the host that invokes it.

    Production callers must set ``workspace`` to a sibling repair worktree
    (see ``repair_worktree.ensure_repair_worktree``), not the live cold-queue
    checkout — ``debug_fix`` runs ``git checkout -B`` in ``workspace``.
    """

    workspace: Path
    agent_bin: str = field(default_factory=lambda: os.environ.get("CURSOR_AGENT_BIN", "agent"))
    model: str | None = field(
        default_factory=lambda: os.environ.get("CURSOR_AGENT_MODEL") or DEFAULT_CURSOR_AGENT_MODEL
    )
    timeout_sec: int = 1800
    create_pr: bool = True
    push: bool = True
    remote: str = "origin"
    base_ref: str = "HEAD"
    # GitHub PR merge target — live queue tip, never master/main.
    pr_base: str = field(
        default_factory=lambda: os.environ.get(
            "BABYSITTER_PR_BASE", "synthesis-snapshot-20260622"
        )
    )
    sandbox: str = "disabled"
    log_emit: LogEmit | None = None
    prompt_extra: str = ""

    def _emit(self, incident: IncidentRecord, marker: str, detail: str = "") -> None:
        if self.log_emit is not None:
            self.log_emit(incident.cell_id, marker, detail)

    def _build_prompt(self, incident: IncidentRecord) -> str:
        parts = [
            "You are the zero-acc babysitter repair agent running in Cursor CLI headless mode.",
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

    def _agent_cmd(self, prompt: str) -> list[str]:
        cmd = [
            self.agent_bin,
            "-p",
            "--force",
            "--trust",
            "--sandbox",
            self.sandbox,
            "--workspace",
            str(self.workspace),
            "--output-format",
            "text",
        ]
        if self.model:
            cmd.extend(["--model", self.model])
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
        msg = f"babysitter CLI repair {incident.incident_id}"
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
            f"Autonomous Cursor CLI repair for incident `{incident.incident_id}`.\n\n"
            f"- cell: `{incident.cell_id}`\n"
            f"- path_kind: `{incident.path_kind}`\n"
            f"- broken_sha: `{incident.broken_sha}`\n"
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
            f"branch={branch} bin={self.agent_bin} model={self.model} attempt={incident.cloud_attempt_count}",
        )
        self._emit(
            incident,
            "CLOUD_AGENT_START",
            f"via=cursor_cli branch={branch}",
        )

        try:
            proc = subprocess.run(
                cmd,
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
                check=False,
                env={**os.environ},
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
                f"CLI_AGENT_FAIL: agent exited {proc.returncode}: {detail!r}"
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
