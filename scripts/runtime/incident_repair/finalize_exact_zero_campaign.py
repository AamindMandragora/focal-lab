#!/usr/bin/env python3
"""Freeze corrected baseline evidence and the approved phased cold queue."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from scripts.runtime import build_full_baseline_cold_manifest as builder
from scripts.runtime import run_cold_synthesis_queue as queue


logger = logging.getLogger("exact-zero-finalizer")

REPAIR_MANIFEST = Path(
    "saved-results/2026-08-04-exact-zero-baseline-repair-manifest.json"
)
MONITOR_REPORT = Path("saved-results/2026-08-04-exact-zero-baseline-monitor.json")
GCD_SUPERSESSIONS = Path(
    "saved-results/2026-08-04-exact-zero-baseline-supersessions.json"
)
RECOVERY_SNAPSHOT = Path(
    "saved-results/2026-08-04-priority-interruption-recovery-snapshot.json"
)
SOURCE_MANIFEST = Path("saved-results/2026-08-03-full-baseline-cold-manifest.json")
CORRECTED_EVIDENCE = Path(
    "saved-results/2026-08-05-corrected-full-baseline-evidence.json"
)
CORRECTED_MANIFEST = Path(
    "saved-results/2026-08-05-corrected-full-baseline-cold-manifest.json"
)
RECOVERY_HISTORY_ROOT = Path("saved-results/2026-08-05-corrected-recovery-history")

REPAIR_V1_ROOT = Path("outputs/baselines/exact-zero-repair-20260804")
GCD_V2_ROOT = Path("outputs/baselines/exact-zero-repair-20260804-gcd-sampling-v2")
V7_ROOT = Path("outputs/baselines/exact-zero-repair-20260805-approved-v7")
V8_ROOT = Path("outputs/baselines/exact-zero-repair-20260805-cache-v8")
GCD_V2_LABEL = "smiles-acrylates-qwen25-1p5b-gcd"

CHANGED_TARGET_CELLS = {
    "spider-qwen35-2b",
    "smiles-acrylates-qwen25-1p5b",
    "smiles-acrylates-qwen25-7b",
    "smiles-acrylates-qwen35-2b",
    "smiles-acrylates-qwen35-4b",
    "smiles-chain_extenders-qwen25-1p5b",
    "smiles-chain_extenders-qwen25-7b",
    "smiles-chain_extenders-qwen35-2b",
    "smiles-chain_extenders-qwen35-4b",
    "smiles-isocyanates-qwen25-1p5b",
    "smiles-isocyanates-qwen25-7b",
    "smiles-isocyanates-qwen35-2b",
    "smiles-isocyanates-qwen35-4b",
}
FULL_MEMORY_RETRY_CELLS = {
    "smiles-acrylates-qwen25-1p5b",
    "smiles-acrylates-qwen25-7b",
}
RECOVERY_CELLS = {"gsm-qwen35-2b", "gsm-qwen35-4b"}
UNCHANGED_NEVER_STARTED_CELLS = {
    "spider-qwen25-1p5b",
    "spider-qwen25-7b",
    "spider-qwen35-4b",
}
HELDOUT_ONLY_CELLS = {
    "gsm-qwen25-1p5b",
    "gsm-qwen25-7b",
    "smiles-acrylates-qwen35-4b",
}
FRESH_CHANGED_CELLS = (
    CHANGED_TARGET_CELLS - FULL_MEMORY_RETRY_CELLS - HELDOUT_ONLY_CELLS
)
APPROVED_NEW_AUTHOR_CALLS = 675


class FinalizationError(ValueError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FinalizationError(f"invalid JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise FinalizationError(f"{path} must contain a JSON object")
    return payload


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def select_replacement_path(repo: Path, label: str, suffix: Path) -> Path:
    roots = [GCD_V2_ROOT] if label == GCD_V2_LABEL else []
    roots.extend([V8_ROOT, V7_ROOT, REPAIR_V1_ROOT])
    for root in roots:
        candidate = repo / root / suffix
        if candidate.is_file():
            return candidate
    raise FinalizationError(f"no accepted replacement artifact for {label}")


def collect_replacement_paths(repo: Path) -> dict[str, Path]:
    repair_manifest = _load_json(repo / REPAIR_MANIFEST)
    entries = (
        repair_manifest.get("rows")
        or repair_manifest.get("entries")
        or repair_manifest.get("labels")
        or []
    )
    if (
        int(repair_manifest.get("source_exact_zero_count", -1)) != 31
        or not isinstance(entries, list)
        or len(entries) != 31
    ):
        raise FinalizationError("repair manifest must contain exactly 31 labels")
    labels = [str(entry.get("label", "")) for entry in entries]
    if len(set(labels)) != 31 or any(not label for label in labels):
        raise FinalizationError(
            "repair manifest labels must be 31 unique nonblank values"
        )

    monitor = _load_json(repo / MONITOR_REPORT)
    accepted_v1 = {
        str(row.get("label")): row
        for row in monitor.get("rows", [])
        if str(row.get("status", "")).startswith("accepted")
    }
    gcd_records = {
        str(row.get("label")): row
        for row in _load_json(repo / GCD_SUPERSESSIONS).get("supersessions", [])
        if str(row.get("status", "")).startswith("accepted")
    }

    selected: dict[str, Path] = {}
    counts = {root: 0 for root in (REPAIR_V1_ROOT, GCD_V2_ROOT, V7_ROOT, V8_ROOT)}
    for entry in entries:
        label = str(entry["label"])
        v1_artifact = Path(str(entry["replacement_artifact"]))
        try:
            suffix = v1_artifact.relative_to(REPAIR_V1_ROOT)
        except ValueError as exc:
            raise FinalizationError(
                f"{label}: repair path escaped the repair root"
            ) from exc
        source = repo / str(entry.get("source_artifact", ""))
        source_sha = str(entry.get("source_sha256", ""))
        if not source.is_file() or _sha256(source) != source_sha:
            raise FinalizationError(f"{label}: frozen exact-zero source hash drifted")
        artifact = select_replacement_path(repo, label, suffix)
        selected[label] = artifact
        relative = artifact.relative_to(repo)
        root = next(
            (candidate for candidate in counts if relative.is_relative_to(candidate)),
            None,
        )
        if root is None:
            raise FinalizationError(
                f"{label}: selected path is outside an approved repair root"
            )
        counts[root] += 1
        if root == REPAIR_V1_ROOT:
            row = accepted_v1.get(label)
            if row is None or row.get("source_sha256") != _sha256(artifact):
                raise FinalizationError(
                    f"{label}: v1 acceptance hash is missing or stale"
                )
        elif root == GCD_V2_ROOT:
            row = gcd_records.get(label)
            if row is None or row.get("replacement_sha256") != _sha256(artifact):
                raise FinalizationError(
                    f"{label}: GCD supersession hash is missing or stale"
                )

    expected = {REPAIR_V1_ROOT: 15, GCD_V2_ROOT: 1, V7_ROOT: 11, V8_ROOT: 4}
    if counts != expected:
        raise FinalizationError(f"replacement priority counts changed: {counts}")
    return selected


def configure_queue(
    base_manifest: dict[str, Any],
    recovery_snapshot: dict[str, Any],
    pinned_csds: dict[str, dict[str, str]],
    *,
    evidence_path: Path,
) -> dict[str, Any]:
    jobs = [copy.deepcopy(job) for job in base_manifest.get("jobs", [])]
    recovery = {
        str(row.get("cell_id")): row
        for row in recovery_snapshot.get("cells", [])
        if str(row.get("cell_id")) in RECOVERY_CELLS
    }
    if set(recovery) != RECOVERY_CELLS:
        raise FinalizationError("recovery snapshot is missing one of the two GSM cells")
    if set(pinned_csds) != HELDOUT_ONLY_CELLS:
        raise FinalizationError("held-out CSD pins must cover exactly three cells")

    for job in jobs:
        cell = str(job["cell_id"])
        job["baseline_source"] = str(evidence_path)
        job["heldout_output_json"] = str(
            Path("outputs/reeval/full_baseline_corrected_20260805") / f"{cell}.json"
        )
        job["interrupted_author_calls"] = 0
        if cell in FRESH_CHANGED_CELLS:
            job.update(queue_phase=1, run_mode="fresh_changed_target")
            job["output_name"] = f"coldq_corrected_20260805_{cell}"
            job["log_file"] = f"outputs/generated/{job['output_name']}/run.log"
        elif cell in FULL_MEMORY_RETRY_CELLS:
            job.update(
                queue_phase=2,
                run_mode="fresh_full_memory_retry",
                requires_exclusive_gpu=True,
            )
            job["output_name"] = f"coldq_corrected_20260805_{cell}"
            job["log_file"] = f"outputs/generated/{job['output_name']}/run.log"
        elif cell in RECOVERY_CELLS:
            row = recovery[cell]
            job.update(
                queue_phase=3,
                run_mode="recovery_remaining_calls",
                max_iterations=int(row["remaining_attempt_cap"]),
                interrupted_author_calls=int(row["started_attempts_charged"]),
                initial_attempt_offset=int(row["started_attempts_charged"]),
                initial_completed_evaluations=int(row["completed_evaluations"]),
            )
            if int(row["completed_evaluations"]):
                job["initial_attempt_history_file"] = str(
                    RECOVERY_HISTORY_ROOT / f"{cell}.json"
                )
        elif cell in UNCHANGED_NEVER_STARTED_CELLS:
            job.update(queue_phase=4, run_mode="unchanged_never_started")
        elif cell in HELDOUT_ONLY_CELLS:
            job.update(queue_phase=5, run_mode="heldout_only", **pinned_csds[cell])
        else:
            raise FinalizationError(f"{cell}: no approved queue phase")

    manifest = {
        "campaign": "full-baseline-corrected-20260805",
        "git_commit": str(base_manifest["git_commit"]),
        "corrected_evidence_path": str(evidence_path),
        "approved_author_call_cap": APPROVED_NEW_AUTHOR_CALLS,
        "planned_author_calls": sum(
            int(job["max_iterations"])
            for job in jobs
            if job["run_mode"] != "heldout_only"
        ),
        "jobs": jobs,
    }
    validate_queue_plan(manifest)
    return manifest


def validate_queue_plan(manifest: dict[str, Any]) -> None:
    jobs = manifest.get("jobs") or []
    by_cell = {str(job.get("cell_id")): job for job in jobs}
    expected_cells = (
        FRESH_CHANGED_CELLS
        | FULL_MEMORY_RETRY_CELLS
        | RECOVERY_CELLS
        | UNCHANGED_NEVER_STARTED_CELLS
        | HELDOUT_ONLY_CELLS
    )
    if len(jobs) != 20 or set(by_cell) != expected_cells:
        raise FinalizationError(
            "corrected queue must contain exactly the approved 20 cells"
        )
    expected_phases = {
        1: FRESH_CHANGED_CELLS,
        2: FULL_MEMORY_RETRY_CELLS,
        3: RECOVERY_CELLS,
        4: UNCHANGED_NEVER_STARTED_CELLS,
        5: HELDOUT_ONLY_CELLS,
    }
    for phase, cells in expected_phases.items():
        actual = {
            cell
            for cell, job in by_cell.items()
            if int(job.get("queue_phase", 0)) == phase
        }
        if actual != cells:
            raise FinalizationError(f"queue phase {phase} changed: {sorted(actual)}")
    planned = sum(
        int(job["max_iterations"])
        for job in jobs
        if job.get("run_mode") != "heldout_only"
    )
    if (
        planned != APPROVED_NEW_AUTHOR_CALLS
        or int(manifest.get("planned_author_calls", -1)) != APPROVED_NEW_AUTHOR_CALLS
        or int(manifest.get("approved_author_call_cap", -1))
        != APPROVED_NEW_AUTHOR_CALLS
    ):
        raise FinalizationError(
            f"corrected queue must schedule exactly {APPROVED_NEW_AUTHOR_CALLS} new author calls"
        )
    if len({str(job["output_name"]) for job in jobs}) != 20:
        raise FinalizationError("corrected queue output names must be unique")
    if len({str(job["heldout_output_json"]) for job in jobs}) != 20:
        raise FinalizationError("corrected held-out output paths must be unique")
    for cell in FULL_MEMORY_RETRY_CELLS:
        if by_cell[cell].get("requires_exclusive_gpu") is not True:
            raise FinalizationError(
                f"{cell}: full-memory retry lost its exclusive GPU gate"
            )
    for cell in RECOVERY_CELLS:
        job = by_cell[cell]
        if int(job["max_iterations"]) + int(job["initial_attempt_offset"]) != 40:
            raise FinalizationError(
                f"{cell}: recovery no longer uses only the original 40 calls"
            )
        if int(job["initial_completed_evaluations"]) > int(
            job["initial_attempt_offset"]
        ):
            raise FinalizationError(
                f"{cell}: restored evaluations exceed charged calls"
            )
    for cell in HELDOUT_ONLY_CELLS:
        job = by_cell[cell]
        if len(str(job.get("heldout_csd_sha256", ""))) != 64:
            raise FinalizationError(f"{cell}: held-out CSD is not hash-pinned")


def validate_launch_approval(
    repo: Path,
    manifest_path: Path,
    approval_path: Path,
) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    validate_queue_plan(manifest)
    if manifest.get("campaign") != "full-baseline-corrected-20260805":
        raise FinalizationError("corrected queue campaign name changed")

    evidence_relative = Path(str(manifest.get("corrected_evidence_path", "")))
    if evidence_relative != CORRECTED_EVIDENCE:
        raise FinalizationError("corrected evidence path changed")
    evidence_path = repo / evidence_relative
    if not evidence_path.is_file() or _sha256(evidence_path) != manifest.get(
        "corrected_evidence_sha256"
    ):
        raise FinalizationError("corrected evidence hash does not match the manifest")

    frozen_inputs = (
        (SOURCE_MANIFEST, "source_manifest_sha256"),
        (RECOVERY_SNAPSHOT, "recovery_snapshot_sha256"),
    )
    for relative, hash_field in frozen_inputs:
        path = repo / relative
        if not path.is_file() or _sha256(path) != manifest.get(hash_field):
            raise FinalizationError(f"frozen input hash changed: {relative}")

    for job in manifest["jobs"]:
        cell = str(job["cell_id"])
        if Path(str(job.get("baseline_source", ""))) != CORRECTED_EVIDENCE:
            raise FinalizationError(f"{cell}: job is not bound to corrected evidence")
        history = str(job.get("initial_attempt_history_file", "")).strip()
        history_sha = str(job.get("initial_attempt_history_sha256", "")).strip()
        if history or history_sha:
            history_path = repo / history
            if (
                not history
                or len(history_sha) != 64
                or not history_path.is_file()
                or _sha256(history_path) != history_sha
            ):
                raise FinalizationError(f"{cell}: recovery history hash changed")
        if job.get("run_mode") == "heldout_only":
            queue.pinned_heldout_csd(job, repo)

    approval = _load_json(approval_path)
    expected_approval = {
        "decision": "approved",
        "reviewer_model": "gpt-5.6-sol",
        "git_commit": str(manifest["git_commit"]),
        "corrected_evidence_sha256": str(manifest["corrected_evidence_sha256"]),
        "queue_manifest_sha256": _sha256(manifest_path),
    }
    mismatches = [
        key
        for key, expected in expected_approval.items()
        if approval.get(key) != expected
    ]
    if mismatches:
        raise FinalizationError(
            "independent launch approval mismatch: " + ", ".join(mismatches)
        )
    return manifest


def _write_recovery_history(repo: Path, snapshot: dict[str, Any]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for row in snapshot.get("cells", []):
        cell = str(row.get("cell_id"))
        completed = int(row.get("completed_evaluations", 0))
        if cell not in RECOVERY_CELLS or completed == 0:
            continue
        progress = repo / str(row["progress_report"])
        expected_sha = str(
            row.get("progress_sha256_before_interrupt")
            or row.get("progress_sha256_after_interrupt")
            or ""
        )
        if not progress.is_file() or _sha256(progress) != expected_sha:
            raise FinalizationError(f"{cell}: frozen progress report hash drifted")
        attempts = _load_json(progress).get("attempts") or []
        if len(attempts) != completed:
            raise FinalizationError(f"{cell}: progress report attempt count changed")
        history = []
        for attempt in attempts:
            evaluation = attempt.get("evaluation") or {}
            history.append(
                {
                    "attempt_number": int(attempt["attempt_number"]),
                    "strategy_code": str(attempt["strategy_code"]),
                    "timestamp": str(attempt.get("timestamp", "restored")),
                    "accuracy": float(evaluation["accuracy"]),
                    "syntax_rate": float(evaluation["syntax_rate"]),
                    "contains_delimiters": bool(evaluation["contains_delimiters"]),
                    "num_examples": int(evaluation["num_examples"]),
                    "num_correct": int(evaluation["num_correct"]),
                    "total_time_seconds": float(
                        evaluation.get("total_time_seconds", 0.0)
                    ),
                }
            )
        output = repo / RECOVERY_HISTORY_ROOT / f"{cell}.json"
        _atomic_json(output, history)
        hashes[cell] = _sha256(output)
    return hashes


def _resolve_pinned_csds(
    repo: Path,
    corrected_jobs: list[dict[str, Any]],
    *,
    launch_commit: str,
) -> dict[str, dict[str, str]]:
    source = _load_json(repo / SOURCE_MANIFEST)
    old_by_cell = {str(job["cell_id"]): job for job in source["jobs"]}
    corrected_by_cell = {str(job["cell_id"]): job for job in corrected_jobs}
    pins: dict[str, dict[str, str]] = {}
    for cell in HELDOUT_ONLY_CELLS:
        old_job = copy.deepcopy(old_by_cell[cell])
        old_job["git_commit"] = str(source["git_commit"])
        old_job["launch_commit"] = launch_commit
        corrected = corrected_by_cell[cell]
        csd = queue.compiled_csd(
            repo,
            str(old_job["output_name"]),
            min_accuracy=float(corrected["min_accuracy"]),
            min_syntax_rate=float(corrected["min_syntax_rate"]),
            job=old_job,
        )
        if csd is None or not csd.is_file():
            raise FinalizationError(
                f"{cell}: no completed CSD qualifies for held-out evaluation"
            )
        pins[cell] = {
            "heldout_csd_path": str(csd.relative_to(repo)),
            "heldout_csd_sha256": _sha256(csd),
        }
    return pins


def build_final_artifacts(repo: Path, git_commit: str) -> tuple[Path, Path]:
    try:
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")
    except ImportError:
        logger.warning("[exact-zero-finalizer] RDKit logging could not be disabled")
    replacements = collect_replacement_paths(repo)
    evidence, base_manifest = builder.build_campaign(
        repo,
        git_commit,
        replacement_paths=replacements,
        evidence_path=CORRECTED_EVIDENCE,
    )
    evidence.update(
        {
            "source_repair_manifest": str(REPAIR_MANIFEST),
            "source_repair_manifest_sha256": _sha256(repo / REPAIR_MANIFEST),
            "replacement_counts": {"repair_v1": 15, "gcd_v2": 1, "v7": 11, "v8": 4},
        }
    )
    _atomic_json(repo / CORRECTED_EVIDENCE, evidence)
    builder.validate_campaign(base_manifest["jobs"], repo)

    snapshot = _load_json(repo / RECOVERY_SNAPSHOT)
    history_hashes = _write_recovery_history(repo, snapshot)
    pins = _resolve_pinned_csds(
        repo,
        base_manifest["jobs"],
        launch_commit=str(snapshot["git_commit"]),
    )
    manifest = configure_queue(
        base_manifest,
        snapshot,
        pins,
        evidence_path=CORRECTED_EVIDENCE,
    )
    manifest.update(
        {
            "corrected_evidence_sha256": _sha256(repo / CORRECTED_EVIDENCE),
            "source_manifest_sha256": _sha256(repo / SOURCE_MANIFEST),
            "recovery_snapshot_sha256": _sha256(repo / RECOVERY_SNAPSHOT),
        }
    )
    for job in manifest["jobs"]:
        cell = str(job["cell_id"])
        if cell in history_hashes:
            job["initial_attempt_history_sha256"] = history_hashes[cell]
        if job.get("run_mode") == "heldout_only":
            queue.pinned_heldout_csd(job, repo)
    validate_queue_plan(manifest)
    _atomic_json(repo / CORRECTED_MANIFEST, manifest)
    logger.warning(
        "[exact-zero-finalizer] wrote evidence=%s manifest=%s replacements=%d calls=%d",
        CORRECTED_EVIDENCE,
        CORRECTED_MANIFEST,
        len(replacements),
        manifest["planned_author_calls"],
    )
    return repo / CORRECTED_EVIDENCE, repo / CORRECTED_MANIFEST


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--git-commit", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
    build_final_artifacts(args.repo.resolve(), str(args.git_commit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
