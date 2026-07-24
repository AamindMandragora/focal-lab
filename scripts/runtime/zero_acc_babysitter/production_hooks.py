"""Production BabysitterHooks: Cursor CLI repair, smoke, merge, kill, recovery.

Callers: production_watch.handle_watch_wake / run_watch_loop; tests.
API: build_production_hooks(...); default kill/merge/smoke/suite/recovery
     implementations with injectable overrides for local sandbox.
User instruction: wire production_watch through Orchestrator to merge.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Callable, Sequence

from scripts.runtime.zero_acc_babysitter.cloud import RepairAgentClient
from scripts.runtime.zero_acc_babysitter.constants import MERGE_BRANCH, SMOKE_MODEL
from scripts.runtime.zero_acc_babysitter.domains import domain_key
from scripts.runtime.zero_acc_babysitter.orchestrator import BabysitterHooks
from scripts.runtime.zero_acc_babysitter.persistence import IncidentRecord
from scripts.runtime.zero_acc_babysitter.repair_worktree import live_head_sha
from scripts.runtime.zero_acc_babysitter.smoke import (
    SmokeDecision,
    SmokeGate,
    SmokeMetrics,
    evaluate_smoke_metrics,
    parse_smoke_report,
    run_hardened_smoke_job,
    smoke_criteria_for,
)
from scripts.runtime.zero_acc_babysitter.suite import (
    TWINS,
    SuiteOutcome,
    classify_tier_a,
)

logger = logging.getLogger("zero-acc-babysitter")

TwinFn = Callable[[str], float]
HelpersFn = Callable[[str], list[str]]
SmokeDecide = Callable[[IncidentRecord], bool]
MergeFn = Callable[[IncidentRecord], str]
KillFn = Callable[[str], None]
RescoreFn = Callable[[str, str], None]
SiblingFn = Callable[[str, str, Sequence[str]], None]
RestartFn = Callable[[str, int], None]
MemoryFn = Callable[[str, int], None]


def twin_name_for_cell(cell_id: str) -> str:
    key = domain_key(cell_id)
    if key == "gsm":
        return TWINS["gsm"]
    if key == "spider":
        return TWINS["spider"]
    if key.startswith("smiles-"):
        return TWINS["smiles"]
    raise ValueError(f"no twin for {cell_id!r}")


def default_stub_twin_accuracy(cell_id: str) -> float:
    """Sandbox / unset-suite default: twin=0 → Tier A harness path.

    Real GPU twin suite overrides via twin_accuracy=... in build_production_hooks.
    """
    _ = twin_name_for_cell(cell_id)
    return 0.0


def default_stub_helper_failures(_cell_id: str) -> list[str]:
    return []


def read_cell_pid(live_repo: Path, cell_id: str) -> int | None:
    pid_path = live_repo / "logs" / "zero_acc_babysitter" / "cells" / f"{cell_id}.pid"
    if not pid_path.is_file():
        return None
    raw = pid_path.read_text(encoding="utf-8").strip()
    if not raw:
        return None
    try:
        return int(raw.split()[0])
    except ValueError:
        return None


def kill_cell_process_group(cell_id: str, *, live_repo: Path) -> None:
    """Kill the cell synthesis process group only (not the whole queue).

    Prefers ``logs/zero_acc_babysitter/cells/{cell_id}.pid``. Falls back to
    ``pgrep -f`` on the cell id. Uses killpg when the pid is a session leader.
    """
    pid = read_cell_pid(live_repo, cell_id)
    if pid is None:
        try:
            proc = subprocess.run(
                ["pgrep", "-f", cell_id],
                capture_output=True,
                text=True,
                check=False,
            )
            for line in (proc.stdout or "").splitlines():
                line = line.strip()
                if line.isdigit():
                    pid = int(line)
                    break
        except OSError as exc:
            logger.warning("pgrep failed cell=%s err=%s", cell_id, exc)
    if pid is None:
        logger.info("kill_cell: no pid for cell=%s", cell_id)
        return
    try:
        os.killpg(pid, signal.SIGTERM)
        logger.info("kill_cell: killpg SIGTERM pid=%s cell=%s", pid, cell_id)
    except ProcessLookupError:
        return
    except PermissionError:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    time.sleep(0.05)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def commit_live_dirty_state(live: Path, incident_id: str) -> None:
    """Commit tracked dirty state on live so a repair merge can proceed.

    git refuses to merge over ANY dirty path it must update — even when the
    working-tree bytes already equal the incoming content (verified on git
    2.34, incident spider-qwen25-1p5b:2:telemetry:1784857356). The live
    checkout routinely carries uncommitted deploy state, so committing it
    (never stashing/discarding) is the only merge-unblock that cannot lose
    operator work; the pre-merge commit also preserves live's side of any
    hunk later resolved toward the repair branch by ``-X theirs``.
    """
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=live,
        capture_output=True,
        text=True,
        check=False,
    )
    if not (status.stdout or "").strip():
        return
    commit = subprocess.run(
        [
            "git",
            "commit",
            "-a",
            "--no-verify",
            "-m",
            f"babysitter: snapshot live deploy state before merging {incident_id}",
        ],
        cwd=live,
        capture_output=True,
        text=True,
        check=False,
    )
    if commit.returncode != 0:
        raise RuntimeError(
            "pre-merge commit of live dirty state failed: "
            f"{(commit.stderr or commit.stdout or '')[:400]!r}"
        )
    logger.info(
        "committed live dirty deploy state before merge (%s)", incident_id
    )


def merge_pr_and_pull(
    incident: IncidentRecord,
    *,
    live_repo: Path,
    merge_branch: str = MERGE_BRANCH,
) -> str:
    """Merge the repair PR into ``merge_branch``, pull on live, return HEAD sha.

    Prefer ``gh pr merge`` when ``incident.pr_url`` looks like a GitHub PR URL;
    otherwise merge the repair branch locally if ``pr_url`` is ``branch:...``.
    """
    live = live_repo.resolve()
    pr_url = (incident.pr_url or "").strip()
    if pr_url.startswith("http") and "/pull/" in pr_url:
        merge = subprocess.run(
            ["gh", "pr", "merge", pr_url, "--merge", "--delete-branch=false"],
            cwd=live,
            capture_output=True,
            text=True,
            check=False,
        )
        if merge.returncode != 0:
            raise RuntimeError(
                f"gh pr merge failed: {(merge.stderr or merge.stdout or '')[:400]!r}"
            )
    elif pr_url.startswith("branch:"):
        branch = pr_url.split(":", 1)[1]
        fetch = subprocess.run(
            ["git", "fetch", "origin", branch],
            cwd=live,
            capture_output=True,
            text=True,
            check=False,
        )
        if fetch.returncode != 0:
            raise RuntimeError(
                f"git fetch origin {branch} failed: {(fetch.stderr or fetch.stdout)!r}"
            )
        checkout = subprocess.run(
            ["git", "checkout", merge_branch],
            cwd=live,
            capture_output=True,
            text=True,
            check=False,
        )
        if checkout.returncode != 0:
            raise RuntimeError(
                f"git checkout {merge_branch} failed: {(checkout.stderr or checkout.stdout)!r}"
            )
        commit_live_dirty_state(live, incident.incident_id)
        # -X theirs: the repair branch is the smoked/gated content, so any
        # divergent hunk resolves toward it; live's side survives in the
        # pre-merge snapshot commit above.
        merged = subprocess.run(
            ["git", "merge", "--no-edit", "-X", "theirs", f"origin/{branch}"],
            cwd=live,
            capture_output=True,
            text=True,
            check=False,
        )
        if merged.returncode != 0:
            subprocess.run(
                ["git", "merge", "--abort"],
                cwd=live,
                capture_output=True,
                text=True,
                check=False,
            )
            raise RuntimeError(
                f"git merge origin/{branch} failed: {(merged.stderr or merged.stdout)!r}"
            )
    else:
        logger.warning(
            "merge_and_pull: no pr_url for %s; pull only", incident.incident_id
        )

    pull = subprocess.run(
        ["git", "pull", "--ff-only", "origin", merge_branch],
        cwd=live,
        capture_output=True,
        text=True,
        check=False,
    )
    if pull.returncode != 0:
        logger.warning(
            "git pull origin %s failed: %s",
            merge_branch,
            (pull.stderr or pull.stdout or "")[:300],
        )
    return live_head_sha(live)



def attach_smoke_report(
    incident: IncidentRecord,
    report_path: Path,
    *,
    process_rc: int = 0,
    uniq_tokens: int | None = None,
) -> SmokeMetrics:
    """Attach a reevaluate JSON to the incident so default_smoke_decide uses UV/Acc gates."""
    incident.extra = dict(incident.extra or {})
    incident.extra["smoke_report_path"] = str(report_path)
    incident.extra["smoke_process_rc"] = int(process_rc)
    incident.extra["smoke_attempt"] = incident.cloud_attempt_count
    if uniq_tokens is not None:
        incident.extra["smoke_uniq_tokens"] = int(uniq_tokens)
    return parse_smoke_report(
        Path(report_path), process_rc=process_rc, uniq_tokens=uniq_tokens
    )


def metrics_from_incident(incident: IncidentRecord) -> SmokeMetrics | None:
    """Build SmokeMetrics from incident.extra when a real smoke report was run."""
    extra = incident.extra or {}
    report = extra.get("smoke_report_path")
    if report:
        return parse_smoke_report(
            Path(str(report)),
            process_rc=extra.get("smoke_process_rc"),
            uniq_tokens=extra.get("smoke_uniq_tokens"),
            accuracy_line_present=extra.get("smoke_accuracy_line_present"),
            oom_present=extra.get("smoke_oom_present"),
            helper_failures_remaining=extra.get("smoke_helper_failures_remaining"),
            twin_accuracy_pct=extra.get("smoke_twin_accuracy_pct"),
        )
    raw = extra.get("smoke_metrics")
    if isinstance(raw, dict):
        return SmokeMetrics(
            accuracy_pct=raw.get("accuracy_pct"),
            unique_valid_count=raw.get("unique_valid_count"),
            unique_valid_rate=raw.get("unique_valid_rate"),
            twin_accuracy_pct=raw.get("twin_accuracy_pct"),
            helper_failures_remaining=list(raw.get("helper_failures_remaining") or []),
            accuracy_line_present=raw.get("accuracy_line_present"),
            oom_present=raw.get("oom_present"),
            uniq_tokens=raw.get("uniq_tokens"),
            process_rc=raw.get("process_rc"),
        )
    return None



def make_smoke_decide(
    *,
    live_repo: Path,
    repair_worktree: Path,
    smoke_job_runner=None,
):
    """Production decide: run hardened smoke job → UV/Acc metrics gate; never soft-PASS."""

    def _decide(incident: IncidentRecord) -> bool | SmokeDecision:
        criteria = smoke_criteria_for(incident.path_kind)
        measured = metrics_from_incident(incident)
        # Metrics stamped by an earlier cloud attempt are stale: reusing them
        # re-fails every later attempt without ever re-running the smoke on
        # the new PR tip (incident spider-qwen25-1p5b:2:telemetry:1784857356).
        if measured is not None and (incident.extra or {}).get(
            "smoke_attempt"
        ) != incident.cloud_attempt_count:
            logger.info(
                "smoke metrics stale cell=%s (from attempt %s, now %s) — rerunning smoke",
                incident.cell_id,
                (incident.extra or {}).get("smoke_attempt"),
                incident.cloud_attempt_count,
            )
            measured = None
        if measured is None:
            env = os.environ.get("BABYSITTER_SMOKE_STUB", "").strip().lower()
            if env == "pass":
                return SmokeDecision(
                    True,
                    "BABYSITTER_SMOKE_STUB=pass (explicit dry-run)",
                    SmokeMetrics(process_rc=0),
                )
            if env == "fail":
                return SmokeDecision(
                    False,
                    "BABYSITTER_SMOKE_STUB=fail",
                    SmokeMetrics(process_rc=None),
                )
            # Real path: execute smoke job (injectable runner for tests).
            return run_hardened_smoke_job(
                incident,
                live_repo=live_repo,
                repair_worktree=repair_worktree,
                runner=smoke_job_runner,
            )
        decision = evaluate_smoke_metrics(
            cell_id=incident.cell_id,
            path_kind=incident.path_kind,
            metrics=measured,
        )
        logger.info(
            "smoke metrics decide cell=%s kind=%s criteria=%s passed=%s reason=%s "
            "uv_count=%s acc=%s uniq_tokens=%s",
            incident.cell_id,
            incident.path_kind,
            criteria.name,
            decision.passed,
            decision.reason,
            measured.unique_valid_count,
            measured.accuracy_pct,
            measured.uniq_tokens,
        )
        return decision

    return _decide


def default_smoke_decide(incident: IncidentRecord) -> bool | SmokeDecision:
    """Prefer measured metrics; env stub only when no metrics attached.

    Soft uniq_tokens alone never PASSes — evaluate_smoke_metrics enforces
    SMILES UV>0 / GSM Acc>0 / helper micros / telemetry line / no OOM.
    ``BABYSITTER_SMOKE_STUB=pass|fail`` is for local orch dry-runs without a report.
    """
    criteria = smoke_criteria_for(incident.path_kind)
    measured = metrics_from_incident(incident)
    if measured is not None:
        decision = evaluate_smoke_metrics(
            cell_id=incident.cell_id,
            path_kind=incident.path_kind,
            metrics=measured,
        )
        logger.info(
            "smoke metrics decide cell=%s kind=%s criteria=%s passed=%s reason=%s "
            "uv_count=%s acc=%s uniq_tokens=%s",
            incident.cell_id,
            incident.path_kind,
            criteria.name,
            decision.passed,
            decision.reason,
            measured.unique_valid_count,
            measured.accuracy_pct,
            measured.uniq_tokens,
        )
        return decision

    # Safety: without measured metrics, do NOT PASS (would soft-merge like UV=0).
    # Local orch dry-runs must set BABYSITTER_SMOKE_STUB=pass explicitly.
    env = os.environ.get("BABYSITTER_SMOKE_STUB", "fail").strip().lower()
    logger.info(
        "smoke stub decide cell=%s kind=%s criteria=%s env=%s model=%s "
        "(no measured metrics on incident; default fail refuses merge)",
        incident.cell_id,
        incident.path_kind,
        criteria.name,
        env,
        SMOKE_MODEL,
    )
    if env == "pass":
        return SmokeDecision(
            True,
            "BABYSITTER_SMOKE_STUB=pass (explicit dry-run)",
            SmokeMetrics(process_rc=0),
        )
    return SmokeDecision(
        False,
        "no measured smoke metrics (refuse merge; set smoke_report_path or STUB=pass)",
        SmokeMetrics(process_rc=None),
    )


def log_rescore_all(cell_id: str, broken_sha: str, *, live_repo: Path) -> None:
    """Record rescore intent; real reevaluate_compiled_csd is focal/GPU work."""
    out = (
        live_repo
        / "logs"
        / "zero_acc_babysitter"
        / "rescore"
        / f"{cell_id}_{broken_sha[:12]}"
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "planned.txt").write_text(
        f"rescore_all cell={cell_id} broken_sha={broken_sha}\n"
        "mode=A-style --max-iterations 1 bars 0 --initial-strategy-file re-eval only\n",
        encoding="utf-8",
    )
    logger.info("rescore_all planned cell=%s sha=%s out=%s", cell_id, broken_sha, out)


def log_sibling_reeval(
    cell_id: str,
    broken_sha: str,
    siblings: Sequence[str],
    *,
    live_repo: Path,
) -> None:
    out = (
        live_repo
        / "logs"
        / "zero_acc_babysitter"
        / "siblings"
        / f"{cell_id}_{broken_sha[:12]}"
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "planned.txt").write_text(
        f"sibling_reeval trigger={cell_id} broken_sha={broken_sha}\n"
        f"siblings={','.join(siblings)}\n",
        encoding="utf-8",
    )
    logger.info(
        "sibling_reeval planned cell=%s siblings=%s sha=%s",
        cell_id,
        list(siblings),
        broken_sha,
    )


def log_restart_from_k(cell_id: str, k: int, *, live_repo: Path) -> None:
    out = live_repo / "logs" / "zero_acc_babysitter" / "restart_k"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{cell_id}.txt").write_text(
        f"restart_from_k cell={cell_id} k={k}\n", encoding="utf-8"
    )
    logger.info("restart_from_k planned cell=%s k=%s", cell_id, k)


def log_memory_resume(cell_id: str, attempt_index: int, *, live_repo: Path) -> None:
    out = live_repo / "logs" / "zero_acc_babysitter" / "memory_resume"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{cell_id}.txt").write_text(
        f"memory_resume cell={cell_id} attempt={attempt_index}\n",
        encoding="utf-8",
    )
    logger.info(
        "memory_resume planned cell=%s attempt=%s", cell_id, attempt_index
    )


def build_production_hooks(
    *,
    live_repo: Path,
    repair_worktree: Path,
    repair_client: RepairAgentClient,
    cell_ids: Sequence[str],
    twin_accuracy: TwinFn | None = None,
    helper_failures: HelpersFn | None = None,
    smoke_decide: SmokeDecide | None = None,
    smoke_job_runner=None,
    merge_and_pull: MergeFn | None = None,
    kill_cell: KillFn | None = None,
    rescore_all: RescoreFn | None = None,
    sibling_reeval: SiblingFn | None = None,
    restart_from_k: RestartFn | None = None,
    memory_resume: MemoryFn | None = None,
    max_cloud_attempts_override: int | None = None,
) -> BabysitterHooks:
    """Build hooks for Orchestrator. Overrides enable local sandbox TDD."""
    live = live_repo.resolve()
    repair = repair_worktree.resolve()
    _ = cell_ids
    _ = max_cloud_attempts_override

    twin_fn = twin_accuracy or default_stub_twin_accuracy
    helpers_fn = helper_failures or default_stub_helper_failures
    decide = smoke_decide or make_smoke_decide(
        live_repo=live,
        repair_worktree=repair,
        smoke_job_runner=smoke_job_runner,
    )
    gate = SmokeGate(decide)

    def _cloud(incident: IncidentRecord) -> str | None:
        if hasattr(repair_client, "workspace"):
            repair_client.workspace = repair  # type: ignore[attr-defined]
        if hasattr(repair_client, "base_ref"):
            repair_client.base_ref = live_head_sha(live)  # type: ignore[attr-defined]
        return repair_client.debug_fix(incident)

    def _smoke(incident: IncidentRecord) -> bool:
        report = gate.evaluate(incident)
        logger.info(
            "smoke evaluate cell=%s kind=%s passed=%s reason=%s cmd=%s metrics=%s",
            incident.cell_id,
            incident.path_kind,
            report.passed,
            report.reason,
            report.command,
            report.metrics,
        )
        if not report.passed:
            # Ensure callers/logs can see why merge is refused.
            incident.extra = dict(incident.extra or {})
            incident.extra["smoke_fail_reason"] = report.reason or "SMOKE_FAIL"
        return report.passed

    def _merge(incident: IncidentRecord) -> str:
        if merge_and_pull is not None:
            return merge_and_pull(incident)
        return merge_pr_and_pull(incident, live_repo=live)

    def _kill(cell_id: str) -> None:
        if kill_cell is not None:
            kill_cell(cell_id)
            return
        kill_cell_process_group(cell_id, live_repo=live)

    def _rescore(cell_id: str, sha: str) -> None:
        if rescore_all is not None:
            rescore_all(cell_id, sha)
            return
        log_rescore_all(cell_id, sha, live_repo=live)

    def _siblings(cell_id: str, sha: str, siblings: Sequence[str]) -> None:
        if sibling_reeval is not None:
            sibling_reeval(cell_id, sha, siblings)
            return
        log_sibling_reeval(cell_id, sha, siblings, live_repo=live)

    def _restart(cell_id: str, k: int) -> None:
        if restart_from_k is not None:
            restart_from_k(cell_id, k)
            return
        log_restart_from_k(cell_id, k, live_repo=live)

    def _memory(cell_id: str, attempt: int) -> None:
        if memory_resume is not None:
            memory_resume(cell_id, attempt)
            return
        log_memory_resume(cell_id, attempt, live_repo=live)

    hooks = BabysitterHooks(
        twin_accuracy=twin_fn,
        helper_failures=helpers_fn,
        run_cloud_debug=_cloud,
        run_smoke=_smoke,
        merge_and_pull=_merge,
        kill_cell=_kill,
        broken_sha=lambda: live_head_sha(live),
        rescore_all=_rescore,
        sibling_reeval=_siblings,
        restart_from_k=_restart,
        memory_resume=_memory,
    )
    if max_cloud_attempts_override is not None:
        # Read by production_watch.build_watch_orchestrator / handle_watch_wake.
        setattr(hooks, "_max_cloud_attempts_override", int(max_cloud_attempts_override))
    return hooks


def suite_outcome_for(
    cell_id: str, *, twin_accuracy: TwinFn, helper_failures: HelpersFn
) -> SuiteOutcome:
    outcome = SuiteOutcome(
        twin_accuracy=float(twin_accuracy(cell_id)),
        helper_failures=list(helper_failures(cell_id)),
    )
    _ = classify_tier_a(outcome)
    return outcome
