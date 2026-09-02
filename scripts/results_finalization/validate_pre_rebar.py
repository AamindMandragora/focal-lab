#!/usr/bin/env python3
"""Validate prerequisite queue state and freeze a results-matrix snapshot hash."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger("results_finalization.validate_pre_rebar")
FIXED_PAID_LABELS = [
    "smiles-qwen35-2b-chain_extenders",
    "smiles-qwen35-4b-acrylates",
    "smiles-qwen35-4b-chain_extenders",
    "smiles-qwen35-9b-acrylates",
    "smiles-qwen35-9b-chain_extenders",
    "gsm14b",
    "spider14b",
]
TERMINAL_STATUSES = {
    "failed",
    "gpu_wait_failed",
    "heldout_failed",
    "heldout_gpu_failed",
    "no_success_csd",
    "ran",
    "skip_exists",
    "synthesis_failed",
}
OUTPUT_STATUSES = {"ran", "skip_exists"}
FAILURE_STATUSES = TERMINAL_STATUSES - OUTPUT_STATUSES


class ValidationError(ValueError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_completion_marker(path: Path, matrix_sha256: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValidationError(f"completion marker does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationError(f"completion marker is invalid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValidationError(f"completion marker must be an object: {path}")
    status = payload.get("status")
    if status != "complete":
        raise ValidationError(f"completion marker is not complete: {path}: status={status!r}")
    if not payload.get("completed_at"):
        raise ValidationError(f"completion marker has no completed_at timestamp: {path}")
    required_labels = payload.get("required_labels", [])
    if not isinstance(required_labels, list) or any(
        not isinstance(label, str) or not label for label in required_labels
    ):
        raise ValidationError(f"completion marker required_labels are invalid: {path}")
    if len(set(required_labels)) != len(required_labels):
        raise ValidationError(f"completion marker required_labels contain duplicates: {path}")
    if required_labels:
        jobs = payload.get("jobs")
        if not isinstance(jobs, dict) or set(jobs) != set(required_labels):
            raise ValidationError(
                f"completion marker jobs do not match required_labels: {path}"
            )
        for label in required_labels:
            row = jobs[label]
            if not isinstance(row, dict) or row.get("label") != label:
                raise ValidationError(
                    f"completion marker has an invalid job row for {label!r}: {path}"
                )
            if row.get("status") not in TERMINAL_STATUSES:
                raise ValidationError(
                    f"completion marker has nonterminal job {label!r}: {path}"
                )
    recorded_matrix = payload.get("matrix_sha256")
    if recorded_matrix is not None and recorded_matrix != matrix_sha256:
        raise ValidationError(
            f"completion marker matrix SHA-256 does not match: {path}"
        )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "completed_at": payload.get("completed_at"),
        "status": status,
        "required_labels": required_labels,
    }


def read_latest_statuses(path: Path) -> tuple[dict[str, dict[str, str]], list[str]]:
    if not path.is_file():
        raise ValidationError(f"status TSV does not exist: {path}")
    latest: dict[str, dict[str, str]] = {}
    order: list[str] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required_columns = {
            "label",
            "status",
            "finished_at",
            "exit_code",
            "output_json",
            "log",
        }
        if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
            raise ValidationError(
                f"status TSV is missing required columns {sorted(required_columns)}: {path}"
            )
        for row in reader:
            label = (row.get("label") or "").strip()
            if not label:
                raise ValidationError(f"status TSV contains an empty label: {path}")
            if label not in latest:
                order.append(label)
            latest[label] = dict(row)
    if not latest:
        raise ValidationError(f"status TSV has no rows: {path}")
    return latest, order


def validate_status_file(
    path: Path,
    *,
    repo: Path,
    required_labels: list[str] | None = None,
) -> dict[str, Any]:
    latest, order = read_latest_statuses(path)
    labels = required_labels if required_labels is not None else order
    missing = [label for label in labels if label not in latest]
    if missing:
        raise ValidationError(
            f"status TSV is missing required labels {missing}: {path}"
        )
    latest_statuses: dict[str, str] = {}
    for label in labels:
        row = latest[label]
        status = (row.get("status") or "").strip()
        if status not in TERMINAL_STATUSES:
            raise ValidationError(
                f"status TSV has nonterminal label {label!r}: status={status!r}"
            )
        latest_statuses[label] = status
        if not (row.get("finished_at") or "").strip():
            raise ValidationError(
                f"terminal status for {label!r} has no finished_at timestamp"
            )
        exit_code = (row.get("exit_code") or "").strip()
        try:
            int(exit_code)
        except ValueError as exc:
            raise ValidationError(
                f"terminal status for {label!r} has invalid exit_code={exit_code!r}"
            ) from exc
        if status in OUTPUT_STATUSES:
            output_text = (row.get("output_json") or "").strip()
            if not output_text:
                raise ValidationError(
                    f"status {status!r} for {label!r} claims a result but output_json is empty"
                )
            output_path = Path(output_text)
            if not output_path.is_absolute():
                output_path = repo / output_path
            if not output_path.is_file() or output_path.stat().st_size == 0:
                raise ValidationError(
                    f"result output does not exist or is empty for {label!r}: {output_path}"
                )
        if status in FAILURE_STATUSES:
            log_text = (row.get("log") or "").strip()
            if not log_text:
                raise ValidationError(f"failure status for {label!r} has no log path")
            log_path = Path(log_text)
            if not log_path.is_absolute():
                log_path = repo / log_path
            if not log_path.is_file() or log_path.stat().st_size == 0:
                raise ValidationError(
                    f"failure log does not exist or is empty for {label!r}: {log_path}"
                )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "latest_statuses": dict(sorted(latest_statuses.items())),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument(
        "--completion-marker", type=Path, action="append", default=[], required=True
    )
    parser.add_argument("--collection-status-tsv", type=Path, action="append", default=[])
    parser.add_argument("--paid-status-tsv", type=Path, action="append", default=[])
    parser.add_argument("--required-paid-label", action="append", default=[])
    parser.add_argument("--snapshot-json", type=Path, required=True)
    return parser.parse_args(argv)


def validate(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.resolve()
    matrix = args.matrix.resolve()
    if not repo.is_dir():
        raise ValidationError(f"repo does not exist: {repo}")
    if not matrix.is_file() or matrix.stat().st_size == 0:
        raise ValidationError(f"results matrix does not exist or is empty: {matrix}")
    matrix_sha256 = sha256_file(matrix)
    markers = [
        load_completion_marker(path.resolve(), matrix_sha256)
        for path in args.completion_marker
    ]
    collection_status = [
        validate_status_file(path.resolve(), repo=repo)
        for path in args.collection_status_tsv
    ]
    if not args.paid_status_tsv:
        raise ValidationError("at least one paid status TSV is required")
    if not args.required_paid_label:
        raise ValidationError("at least one required paid label is required")
    if len(set(args.required_paid_label)) != len(args.required_paid_label):
        raise ValidationError("required paid labels contain duplicates")
    if set(args.required_paid_label) != set(FIXED_PAID_LABELS):
        raise ValidationError(
            "required paid labels must be exactly the seven fixed queue labels: "
            + ", ".join(FIXED_PAID_LABELS)
        )
    if not any(
        set(marker["required_labels"]) == set(FIXED_PAID_LABELS)
        for marker in markers
    ):
        raise ValidationError(
            "completion markers do not include the exact fixed seven-job queue marker"
        )
    paid_status = [
        validate_status_file(
            path.resolve(), repo=repo, required_labels=FIXED_PAID_LABELS
        )
        for path in args.paid_status_tsv
    ]
    return {
        "state": "complete",
        "validated_at": utc_now(),
        "repo": str(repo),
        "matrix": str(matrix),
        "matrix_sha256": matrix_sha256,
        "completion_markers": markers,
        "collection_status": collection_status,
        "paid_status": paid_status,
        "required_paid_labels": FIXED_PAID_LABELS,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args.snapshot_json.unlink(missing_ok=True)
    LOGGER.info(
        "[pre-rebar-validator] matrix=%s markers=%d collection_status=%d paid_status=%d",
        args.matrix,
        len(args.completion_marker),
        len(args.collection_status_tsv),
        len(args.paid_status_tsv),
    )
    try:
        snapshot = validate(args)
        args.snapshot_json.parent.mkdir(parents=True, exist_ok=True)
        args.snapshot_json.write_text(
            json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except (OSError, ValidationError) as exc:
        args.snapshot_json.unlink(missing_ok=True)
        LOGGER.error("[pre-rebar-validator] %s", exc)
        return 2
    LOGGER.info(
        "[pre-rebar-validator] complete matrix_sha=%s snapshot=%s",
        snapshot["matrix_sha256"],
        args.snapshot_json,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
