#!/usr/bin/env python3
"""Review exact-zero baseline repairs every five minutes and block synthesis."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from scripts.runtime import build_full_baseline_cold_manifest as campaign_builder


LOGGER = logging.getLogger("exact-zero-baseline-monitor")
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = Path(
    "saved-results/2026-08-04-exact-zero-baseline-repair-manifest.json"
)
DEFAULT_REPORT = Path(
    "saved-results/2026-08-04-exact-zero-baseline-monitor.json"
)
DEFAULT_BLOCK = Path(".context/exact-zero-repair-synthesis.blocked")
DEFAULT_ACCEPTANCES = Path(
    "saved-results/2026-08-04-exact-zero-baseline-acceptances.json"
)
DEFAULT_REPAIR_OUTPUT_ROOT = Path(
    "outputs/baselines/exact-zero-repair-20260804"
)
ACCEPTED_STATUSES = {"accepted", "accepted_reviewed_zero"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _cell_context(cell_id: str) -> tuple[str, int, str | None]:
    if cell_id.startswith("spider-"):
        return "spider", 300, None
    if cell_id.startswith("gsm-"):
        return "gsm_symbolic", 49, None
    if cell_id.startswith("smiles-") and "-qwen" in cell_id:
        smiles_class = cell_id[len("smiles-") :].rsplit("-qwen", 1)[0]
        if smiles_class:
            return "smiles", 50, smiles_class
    raise ValueError(f"unsupported repair cell_id: {cell_id}")


def _answer_diagnostics(raw: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {
            "nonblank_answer_count": None,
            "unique_generated_answer_count": None,
            "answer_sample_sha256": [],
        }
    answers = payload.get("answers") if isinstance(payload, dict) else None
    if not isinstance(answers, list):
        return {
            "nonblank_answer_count": None,
            "unique_generated_answer_count": None,
            "answer_sample_sha256": [],
        }
    generated = [
        str(row.get("generated_answer", "")).strip()
        for row in answers
        if isinstance(row, dict)
    ]
    nonblank = [answer for answer in generated if answer]
    return {
        "nonblank_answer_count": len(nonblank),
        "unique_generated_answer_count": len(set(nonblank)),
        "answer_sample_sha256": [
            hashlib.sha256(answer.encode("utf-8")).hexdigest()
            for answer in nonblank[:3]
        ],
    }


def _log_diagnostics(repo: Path, campaign: str, label: str) -> dict[str, Any]:
    log_path = repo / "logs" / campaign / f"{label}.log"
    if not log_path.is_file():
        return {"log_file": str(log_path.relative_to(repo)), "log_sha256": None}
    tail = log_path.read_bytes()[-65536:].decode("utf-8", "replace")
    alert_terms = (
        "traceback",
        "error",
        "cuda out of memory",
        "free memory on device",
        "parser",
        "fallback",
        "timeout",
    )
    alerts = [
        line[-500:]
        for line in tail.splitlines()
        if any(term in line.lower() for term in alert_terms)
    ][-20:]
    return {
        "log_file": str(log_path.relative_to(repo)),
        "log_sha256": _sha256(log_path),
        "log_alert_lines": alerts,
    }


def _read_frozen_source_scores(
    raw: bytes,
    path: Path,
    total: int,
) -> tuple[int, int]:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid frozen source {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: frozen source must be an object")
    answers = payload.get("answers")
    if not isinstance(answers, list) or len(answers) != total:
        raise ValueError(f"{path}: frozen source answer count must be {total}")
    if any(
        not isinstance(row, dict) or "generated_answer" not in row
        for row in answers
    ):
        raise ValueError(
            f"{path}: every frozen source row must contain generated_answer"
        )
    metrics = payload.get("metrics") or {}
    if not isinstance(metrics, dict) or int(metrics.get("num_examples") or -1) != total:
        raise ValueError(f"{path}: frozen source metrics.num_examples must be {total}")
    for field in ("accuracy", "syntax_rate"):
        value = payload.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or float(value) != 0.0
        ):
            raise ValueError(
                f"{path}: frozen source {field} must be literal numeric zero"
            )
    return 0, 0


def review_artifact(
    repo: Path,
    campaign: str,
    row: Mapping[str, Any],
    *,
    zero_acceptances: Mapping[str, str],
) -> dict[str, Any]:
    label = str(row["label"])
    relative = Path(str(row["replacement_artifact"]))
    artifact = repo / relative
    base: dict[str, Any] = {
        "label": label,
        "cell_id": str(row["cell_id"]),
        "strategy": str(row["strategy"]),
        "artifact": str(relative),
        "artifact_preserved": True,
    }
    if not artifact.is_file():
        return {**base, "status": "pending", "reason": "replacement artifact missing"}

    raw = artifact.read_bytes()
    digest = _sha256_bytes(raw)
    diagnostics = {
        "source_sha256": digest,
        **_answer_diagnostics(raw),
        **_log_diagnostics(repo, campaign, label),
    }
    try:
        dataset, total, smiles_class = _cell_context(str(row["cell_id"]))
        measured = campaign_builder._read_baseline_bytes(
            raw,
            artifact,
            total,
            str(row["strategy"]),
            repo,
            dataset=dataset,
            smiles_class=smiles_class,
        )
    except (campaign_builder.CampaignError, ValueError) as exc:
        LOGGER.error(
            "[exact-zero-monitor] quarantine label=%s sha256=%s reason=%s",
            label,
            digest,
            exc,
        )
        return {
            **base,
            **diagnostics,
            "status": "quarantined_system_failure",
            "reason": str(exc),
        }

    result = {**base, **diagnostics, **measured}
    if measured["num_correct"] == 0 and measured["syntax_count"] == 0:
        if zero_acceptances.get(label) == digest:
            LOGGER.warning(
                "[exact-zero-monitor] accept reviewed zero label=%s sha256=%s "
                "nonblank=%s unique=%s",
                label,
                digest,
                measured["nonblank_answer_count"],
                measured["unique_generated_answer_count"],
            )
            return {
                **result,
                "status": "accepted_reviewed_zero",
                "reason": "exact artifact hash accepted after skeptical review",
            }
        LOGGER.warning(
            "[exact-zero-monitor] skeptical review required label=%s sha256=%s "
            "nonblank=%s unique=%s",
            label,
            digest,
            measured["nonblank_answer_count"],
            measured["unique_generated_answer_count"],
        )
        return {
            **result,
            "status": "needs_skeptical_review",
            "reason": "functional-looking exact 0/0 requires hash-bound review",
        }
    return {**result, "status": "accepted", "reason": "structural review passed"}


def review_manifest(
    repo: Path,
    manifest_path: Path,
    *,
    zero_acceptances: Mapping[str, str],
    expected_rows: int | None = None,
) -> dict[str, Any]:
    manifest_raw = manifest_path.read_bytes()
    payload = json.loads(manifest_raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{manifest_path}: manifest must be an object")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{manifest_path}: rows must be a non-empty list")
    declared_count = payload.get("source_exact_zero_count")
    if isinstance(declared_count, bool) or not isinstance(declared_count, int):
        raise ValueError(f"{manifest_path}: source_exact_zero_count must be an integer")
    if expected_rows is not None and declared_count != expected_rows:
        raise ValueError(
            f"{manifest_path}: declares {declared_count} exact-zero rows; "
            f"expected {expected_rows}"
        )
    if len(rows) != declared_count:
        raise ValueError(
            f"{manifest_path}: has {len(rows)} rows; declared {declared_count}"
        )
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{manifest_path}: every row must be an object")
    labels = [str(row.get("label") or "") for row in rows]
    if any(not label for label in labels) or len(set(labels)) != len(labels):
        raise ValueError(f"{manifest_path}: row labels must be non-empty and unique")

    repo_root = repo.resolve()

    def resolved_repo_path(value: Any, field: str) -> Path:
        raw = str(value or "")
        if not raw:
            raise ValueError(f"{manifest_path}: {field} is required")
        resolved = (repo_root / Path(raw)).resolve()
        try:
            resolved.relative_to(repo_root)
        except ValueError as exc:
            raise ValueError(f"{manifest_path}: {field} must stay under the repository") from exc
        return resolved

    repair_root = resolved_repo_path(
        payload.get("repair_output_root"),
        "repair_output_root",
    )
    configured_repair_root = (repo_root / DEFAULT_REPAIR_OUTPUT_ROOT).resolve()
    if repair_root != configured_repair_root:
        raise ValueError(
            f"{manifest_path}: repair output root must be {DEFAULT_REPAIR_OUTPUT_ROOT}"
        )
    source_paths: set[Path] = set()
    replacement_paths: set[Path] = set()
    for row in rows:
        label = str(row["label"])
        source = resolved_repo_path(row.get("source_artifact"), f"{label}.source_artifact")
        replacement = resolved_repo_path(
            row.get("replacement_artifact"),
            f"{label}.replacement_artifact",
        )
        if replacement != repair_root and repair_root not in replacement.parents:
            raise ValueError(
                f"{manifest_path}: {label} replacement is outside the repair output root"
            )
        if source in source_paths or replacement in replacement_paths:
            raise ValueError(f"{manifest_path}: source and replacement paths must be unique")
        source_paths.add(source)
        replacement_paths.add(replacement)
        if source_paths & replacement_paths:
            raise ValueError(
                f"{manifest_path}: source and replacement paths must not overlap"
            )
        if not source.is_file():
            raise ValueError(f"{manifest_path}: frozen source is missing for {label}")
        source_raw = source.read_bytes()
        expected_source_sha = str(row.get("source_sha256") or "")
        if len(expected_source_sha) != 64 or _sha256_bytes(source_raw) != expected_source_sha:
            raise ValueError(f"{manifest_path}: source SHA-256 mismatch for {label}")
        _, total, _ = _cell_context(str(row["cell_id"]))
        num_correct, syntax_count = _read_frozen_source_scores(source_raw, source, total)
        if num_correct != 0 or syntax_count != 0:
            raise ValueError(f"{manifest_path}: frozen source is not exact 0/0 for {label}")
    manifest_digest = _sha256_bytes(manifest_raw)
    campaign = str(payload.get("campaign") or "exact-zero-repair-20260804")
    reviewed = [
        review_artifact(
            repo,
            campaign,
            row,
            zero_acceptances=zero_acceptances,
        )
        for row in rows
    ]
    counts = {
        "accepted": sum(row["status"] in ACCEPTED_STATUSES for row in reviewed),
        "pending": sum(row["status"] == "pending" for row in reviewed),
        "needs_skeptical_review": sum(
            row["status"] == "needs_skeptical_review" for row in reviewed
        ),
        "quarantined_system_failure": sum(
            row["status"] == "quarantined_system_failure" for row in reviewed
        ),
    }
    baseline_review_complete = counts["accepted"] == len(rows)
    return {
        "campaign": campaign,
        "checked_at": utc_now(),
        "manifest": str(manifest_path.relative_to(repo)),
        "manifest_sha256": manifest_digest,
        "expected_rows": declared_count,
        "counts": counts,
        "baseline_review_complete": baseline_review_complete,
        "synthesis_blocked": True,
        "block_reason": (
            "awaiting corrected evidence and queue validation"
            if baseline_review_complete
            else "baseline repairs are incomplete or need review"
        ),
        "rows": reviewed,
    }


def write_report_and_gate(report: Mapping[str, Any], report_path: Path, block_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_name(f".{report_path.name}.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(report_path)
    if bool(report["synthesis_blocked"]):
        block_path.parent.mkdir(parents=True, exist_ok=True)
        block_temporary = block_path.with_name(f".{block_path.name}.tmp")
        block_temporary.write_text(
            json.dumps(
                {
                    "checked_at": report["checked_at"],
                    "counts": report["counts"],
                    "reason": report["block_reason"],
                    "report": str(report_path),
                    "manifest_sha256": report.get("manifest_sha256"),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        block_temporary.replace(block_path)


def load_zero_acceptances(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("acceptances") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        raise ValueError(f"{path}: acceptances must be a list")
    accepted: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: every acceptance must be an object")
        label = str(entry.get("label") or "")
        digest = str(entry.get("source_sha256") or "")
        review_reason = str(entry.get("review_reason") or "").strip()
        if not label or len(digest) != 64:
            raise ValueError(f"{path}: invalid hash-bound acceptance")
        if not review_reason:
            raise ValueError(f"{path}: every acceptance requires a review reason")
        accepted[label] = digest
    return accepted


def process_alive(pid: int | None) -> bool | None:
    if pid is None:
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--block-file", type=Path, default=DEFAULT_BLOCK)
    parser.add_argument("--zero-acceptances", type=Path, default=DEFAULT_ACCEPTANCES)
    parser.add_argument("--pool-pid", type=int)
    parser.add_argument("--poll-seconds", type=float, default=300.0)
    parser.add_argument("--expected-rows", type=int, default=31)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def _under(repo: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo / path


def _display_repo_path(repo: Path, path: Path) -> str:
    try:
        return str(path.relative_to(repo))
    except ValueError:
        return str(path)


def poll_once(
    *,
    repo: Path,
    manifest_path: Path,
    report_path: Path,
    block_path: Path,
    acceptance_path: Path,
    expected_rows: int,
    pool_pid: int | None,
) -> dict[str, Any]:
    try:
        report = review_manifest(
            repo,
            manifest_path,
            zero_acceptances=load_zero_acceptances(acceptance_path),
            expected_rows=expected_rows,
        )
    except (OSError, json.JSONDecodeError, ValueError, TypeError, KeyError) as exc:
        manifest_digest: str | None = None
        try:
            if manifest_path.is_file():
                manifest_digest = _sha256(manifest_path)
        except OSError:
            pass
        report = {
            "campaign": "unknown",
            "checked_at": utc_now(),
            "manifest": _display_repo_path(repo, manifest_path),
            "manifest_sha256": manifest_digest,
            "expected_rows": expected_rows,
            "counts": {
                "accepted": 0,
                "pending": expected_rows,
                "needs_skeptical_review": 0,
                "quarantined_system_failure": 0,
            },
            "baseline_review_complete": False,
            "synthesis_blocked": True,
            "block_reason": "monitor input validation failed",
            "monitor_error": f"{type(exc).__name__}: {exc}",
            "rows": [],
        }
        LOGGER.exception(
            "[exact-zero-monitor] fail closed manifest=%s acceptance=%s error=%s",
            manifest_path,
            acceptance_path,
            exc,
        )
    report["pool_pid"] = pool_pid
    report["pool_alive"] = process_alive(pool_pid)
    write_report_and_gate(report, report_path, block_path)
    LOGGER.info(
        "[exact-zero-monitor] poll counts=%s pool_alive=%s synthesis_blocked=%s "
        "monitor_error=%s",
        report["counts"],
        report["pool_alive"],
        report["synthesis_blocked"],
        report.get("monitor_error"),
    )
    return report


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    repo = args.repo.resolve()
    manifest = _under(repo, args.manifest)
    report_path = _under(repo, args.report)
    block_path = _under(repo, args.block_file)
    acceptance_path = _under(repo, args.zero_acceptances)
    while True:
        poll_once(
            repo=repo,
            manifest_path=manifest,
            report_path=report_path,
            block_path=block_path,
            acceptance_path=acceptance_path,
            expected_rows=args.expected_rows,
            pool_pid=args.pool_pid,
        )
        if args.once:
            return 0
        time.sleep(max(1.0, args.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
