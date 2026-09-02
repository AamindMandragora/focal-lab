#!/usr/bin/env python3
"""Build a hash-bound metaDecode re-bar audit and candidate manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger("results_finalization.scan_metadecode_rebar")
SOURCE_REF_HASH_LENGTH = 12
COHORT_FIELDS = (
    "board",
    "dataset",
    "model",
    "class",
    "metric_version",
    "eval_split",
    "prompt_profile",
    "scorer_version",
    "token_budget",
    "max_steps",
    "evaluation_protocol",
)
CANDIDATE_FIELDS = (
    "cell_id",
    "dataset",
    "model",
    "class",
    "target_n",
    "min_accuracy_count",
    "min_syntax_count",
    "recipe_json",
)
REQUIRED_RECIPE_FIELDS = {
    "output_name",
    "heldout_output_json",
    "gpu",
    "gpu_mem_util",
    "gpu_wait_max_used_mib",
    "train_sample_size",
    "eval_max_steps",
    "eval_max_seconds",
    "train_split_name",
    "heldout_split_name",
    "heldout_sample_size",
}
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
SEPARATOR_CELL_RE = re.compile(r"^:?-{3,}:?$")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def validate_pre_rebar_snapshot(snapshot_path: Path, matrix_sha256: str) -> str:
    if not snapshot_path.is_file():
        raise ValueError(f"pre-rebar snapshot is missing: {snapshot_path}")
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("state") != "complete":
        raise ValueError("pre-rebar snapshot is not complete")
    if payload.get("matrix_sha256") != matrix_sha256:
        raise ValueError("pre-rebar snapshot matrix SHA-256 does not match the matrix")
    return sha256_file(snapshot_path)


def _is_table_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 3


def _is_separator_line(line: str) -> bool:
    if not _is_table_line(line):
        return False
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return bool(cells) and all(SEPARATOR_CELL_RE.fullmatch(cell) for cell in cells)


def _table_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _archived_dataset_heading(heading: str) -> bool:
    normalized = heading.casefold()
    return (
        normalized.startswith("gsm-symbolic")
        or normalized == "spider"
        or normalized.startswith("smiles")
    )


def _line_contexts(lines: list[str]) -> list[dict[str, str]]:
    contexts: list[dict[str, str]] = []
    board = ""
    board_heading = ""
    dataset_heading = ""
    class_heading = ""

    for line in lines:
        match = HEADING_RE.match(line)
        if match:
            level = len(match.group(1))
            heading = match.group(2).strip()
            lowered = heading.casefold()
            if level == 2 and "qwen3.5" in lowered and "active" in lowered:
                board = "active_qwen3.5"
                board_heading = heading
                dataset_heading = ""
                class_heading = ""
            elif level == 2 and "qwen2.5" in lowered and "archived" in lowered:
                board = "archived_qwen2.5"
                board_heading = heading
                dataset_heading = ""
                class_heading = ""
            elif board == "active_qwen3.5":
                if level == 2:
                    board = ""
                    board_heading = ""
                    dataset_heading = ""
                    class_heading = ""
                elif level == 3:
                    dataset_heading = heading
                    class_heading = ""
                elif level > 3:
                    class_heading = heading
            elif board == "archived_qwen2.5":
                if level == 2 and _archived_dataset_heading(heading):
                    dataset_heading = heading
                    class_heading = ""
                elif level == 2:
                    board = ""
                    board_heading = ""
                    dataset_heading = ""
                    class_heading = ""
                elif level == 3:
                    class_heading = heading

        contexts.append(
            {
                "board": board,
                "board_heading": board_heading,
                "dataset_heading": dataset_heading,
                "class_heading": class_heading,
            }
        )
    return contexts


def catalog_matrix_rows(matrix_path: Path) -> list[dict[str, Any]]:
    """Catalog table data rows in the active and archived Qwen boards."""

    text = matrix_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    contexts = _line_contexts(lines)
    catalog: list[dict[str, Any]] = []
    index = 0
    while index + 1 < len(lines):
        if _is_table_line(lines[index]) and _is_separator_line(lines[index + 1]):
            headers = _table_cells(lines[index])
            row_index = index + 2
            while row_index < len(lines) and _is_table_line(lines[row_index]):
                context = contexts[row_index]
                if context["board"]:
                    raw_line = lines[row_index]
                    values = _table_cells(raw_line)
                    line_hash = sha256_bytes(raw_line.encode("utf-8"))
                    catalog.append(
                        {
                            "line": row_index + 1,
                            "line_sha256": line_hash,
                            **context,
                            "raw_line": raw_line,
                            "columns": dict(zip(headers, values)),
                            "source_ref": (
                                f"L{row_index + 1}:"
                                f"{line_hash[:SOURCE_REF_HASH_LENGTH]}"
                            ),
                        }
                    )
                row_index += 1
            index = row_index
        else:
            index += 1
    return catalog


def _clean_markdown_cell(value: str) -> str:
    return re.sub(r"[*_~`]", "", value).strip()


def _column_value(columns: dict[str, str], names: tuple[str, ...]) -> tuple[str, str] | None:
    expected = {name.casefold() for name in names}
    for heading, value in columns.items():
        if _clean_markdown_cell(heading).casefold() in expected:
            return heading, value
    return None


def _displayed_ratio(value: str) -> tuple[Decimal, Decimal] | None:
    cleaned = _clean_markdown_cell(value)
    if "→" in cleaned:
        cleaned = cleaned.rsplit("→", 1)[1].strip()
    matches = list(re.finditer(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(%)?", cleaned))
    if not matches:
        return None
    match = matches[0]
    number_text = match.group(1)
    try:
        number = Decimal(number_text)
    except InvalidOperation:
        return None
    decimal_places = len(number_text.partition(".")[2])
    unit = Decimal(1).scaleb(-decimal_places)
    if match.group(2):
        number /= 100
        unit /= 100
    return number, unit / 2


def _source_display_errors(
    catalog_row: dict[str, Any], source: dict[str, Any], label: str
) -> list[str]:
    columns = catalog_row.get("columns", {})
    errors: list[str] = []
    n_column = _column_value(columns, ("N",))
    accuracy_column = _column_value(columns, ("Acc", "UV"))
    syntax_column = _column_value(columns, ("Syntax", "Validity"))
    if n_column is None or accuracy_column is None or syntax_column is None:
        return [f"{label} row does not expose Acc/UV, Syntax/Validity, and N columns"]
    displayed_n = _clean_markdown_cell(n_column[1])
    if not re.fullmatch(r"\d+", displayed_n):
        errors.append(f"{label} has non-canonical displayed N {n_column[1]!r}")
    elif int(displayed_n) != source["n"]:
        errors.append(
            f"{label}.n={source['n']} does not match displayed N={displayed_n}"
        )
    for metric_name, count_name, column in (
        (accuracy_column[0], "accuracy_count", accuracy_column[1]),
        (syntax_column[0], "syntax_count", syntax_column[1]),
    ):
        displayed = _displayed_ratio(column)
        if displayed is None:
            errors.append(f"{label} cannot verify displayed {metric_name} {column!r}")
            continue
        ratio, half_unit = displayed
        actual = Decimal(source[count_name]) / Decimal(source["n"])
        if abs(actual - ratio) > half_unit + Decimal("1e-12"):
            errors.append(
                f"{label}.{count_name}={source[count_name]}/{source['n']} "
                f"does not match displayed {metric_name}={column!r}"
            )
    return errors


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _fraction_summary(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    count_key = f"{metric}_count"
    maximum = max(Fraction(row[count_key], row["n"]) for row in rows)
    winners = [row for row in rows if Fraction(row[count_key], row["n"]) == maximum]
    return {
        "fraction": str(maximum),
        "strategies": [row["strategy"] for row in winners],
        "source_refs": [row["source_ref"] for row in winners],
    }


def _ceil_fraction_times(value: Fraction, multiplier: int) -> int:
    numerator = value.numerator * multiplier
    return (numerator + value.denominator - 1) // value.denominator


def _recipe_json(cell: dict[str, Any]) -> str:
    recipe = cell.get("recipe_json")
    if isinstance(recipe, str):
        parsed = json.loads(recipe)
        if not isinstance(parsed, dict):
            raise ValueError("recipe_json string must encode a JSON object")
        recipe_object = parsed
        serialized = recipe
    elif isinstance(recipe, dict):
        recipe_object = recipe
        serialized = json.dumps(recipe, separators=(",", ":"), sort_keys=True)
    else:
        raise ValueError("recipe_json must be an object or an object-encoded JSON string")
    missing = sorted(REQUIRED_RECIPE_FIELDS - recipe_object.keys())
    if missing:
        raise ValueError(f"recipe_json is missing required fields: {missing}")
    warm_text = json.dumps(recipe_object, sort_keys=True)
    if "initial_strategy" in warm_text or "--initial-strategy-file" in warm_text:
        raise ValueError("recipe_json contains a forbidden warm-start field")
    if recipe_object.get("cold") is not True:
        raise ValueError("recipe_json must explicitly record cold=true")
    if cell.get("dataset") == "smiles" and not recipe_object.get("smiles_task"):
        raise ValueError("SMILES recipe_json requires smiles_task")
    return serialized


def scan_reviewed_matrix(
    matrix_path: Path, reviewed_path: Path, snapshot_path: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    matrix_sha256 = sha256_file(matrix_path)
    snapshot_sha256 = validate_pre_rebar_snapshot(snapshot_path, matrix_sha256)
    reviewed = json.loads(reviewed_path.read_text(encoding="utf-8"))
    catalog = catalog_matrix_rows(matrix_path)
    catalog_by_ref = {row["source_ref"]: row for row in catalog}
    errors: list[str] = []
    cells_audit: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    used_refs: dict[str, str] = {}
    ref_groups: dict[str, list[str]] = {
        "baseline": [],
        "metadecode": [],
        "excluded": [],
    }

    reviewed_sha = reviewed.get("matrix_sha256") if isinstance(reviewed, dict) else None
    if reviewed_sha != matrix_sha256:
        errors.append(
            "reviewed matrix_sha256 does not match the authoritative matrix SHA-256 "
            f"({reviewed_sha!r} != {matrix_sha256!r})"
        )

    cells = reviewed.get("cells", []) if isinstance(reviewed, dict) else []
    if not isinstance(cells, list):
        errors.append("reviewed cells must be a list")
        cells = []

    canonical_cohorts: dict[tuple[Any, ...], str] = {}
    seen_cell_ids: set[str] = set()
    for cell_index, cell in enumerate(cells):
        cell_errors: list[str] = []
        if not isinstance(cell, dict):
            errors.append(f"cells[{cell_index}] must be an object")
            continue
        cell_id = cell.get("cell_id")
        if not isinstance(cell_id, str) or not cell_id.strip():
            cell_errors.append("cell_id must be a nonempty string")
            cell_id = f"cells[{cell_index}]"
        elif cell_id in seen_cell_ids:
            cell_errors.append(f"duplicate cell_id {cell_id!r}; cell identity is ambiguous")
        else:
            seen_cell_ids.add(cell_id)

        for field in COHORT_FIELDS:
            if field not in cell:
                cell_errors.append(f"missing canonical cohort field {field!r}")
            elif field != "class" and (
                cell[field] is None or (isinstance(cell[field], str) and not cell[field].strip())
            ):
                cell_errors.append(f"canonical cohort field {field!r} must be nonempty")

        if all(field in cell for field in COHORT_FIELDS):
            cohort_key = tuple(cell[field] for field in COHORT_FIELDS)
            previous = canonical_cohorts.get(cohort_key)
            if previous is not None:
                cell_errors.append(
                    f"canonical cohort duplicates cell {previous!r}; cohort is ambiguous"
                )
            else:
                canonical_cohorts[cohort_key] = cell_id

        target_n = cell.get("target_n")
        if not _is_int(target_n) or target_n <= 0:
            cell_errors.append("target_n must be a positive integer")
            target_n = 1
        allow_cross_n = cell.get("allow_cross_n")
        if not isinstance(allow_cross_n, bool):
            cell_errors.append("allow_cross_n must be a boolean")
            allow_cross_n = False
        cross_n_reason = cell.get("cross_n_reason", "")
        if allow_cross_n and (
            not isinstance(cross_n_reason, str) or not cross_n_reason.strip()
        ):
            cell_errors.append("allow_cross_n=true requires a nonempty cross_n_reason")

        validated_groups: dict[str, list[dict[str, Any]]] = {
            "baseline": [],
            "metadecode": [],
        }
        contexts: set[tuple[str, str, str]] = set()
        for json_key, group_name in (
            ("baseline_rows", "baseline"),
            ("metadecode_rows", "metadecode"),
        ):
            rows = cell.get(json_key)
            if not isinstance(rows, list):
                cell_errors.append(f"{json_key} must be a list")
                continue
            if group_name == "baseline" and not rows:
                cell_errors.append("baseline_rows must contain at least one source")
            for row_index, source in enumerate(rows):
                label = f"{json_key}[{row_index}]"
                if not isinstance(source, dict):
                    cell_errors.append(f"{label} must be an object")
                    continue
                source_ref = source.get("source_ref")
                if not isinstance(source_ref, str) or source_ref not in catalog_by_ref:
                    cell_errors.append(f"{label} has unknown or stale source_ref {source_ref!r}")
                    continue
                if source_ref in used_refs:
                    cell_errors.append(
                        f"source_ref {source_ref} is used more than once "
                        f"({used_refs[source_ref]} and {cell_id}:{label})"
                    )
                    continue
                used_refs[source_ref] = f"{cell_id}:{label}"
                ref_groups[group_name].append(source_ref)
                catalog_row = catalog_by_ref[source_ref]
                contexts.add(
                    (
                        catalog_row["board"],
                        catalog_row["dataset_heading"],
                        catalog_row["class_heading"],
                    )
                )
                if cell.get("board") != catalog_row["board"]:
                    cell_errors.append(
                        f"{label} belongs to board {catalog_row['board']!r}, "
                        f"not reviewed board {cell.get('board')!r}"
                    )
                strategy = source.get("strategy")
                if not isinstance(strategy, str) or not strategy.strip():
                    cell_errors.append(f"{label}.strategy must be a nonempty string")
                n = source.get("n")
                accuracy_count = source.get("accuracy_count")
                syntax_count = source.get("syntax_count")
                if not _is_int(n) or n <= 0:
                    cell_errors.append(f"{label}.n must be a positive integer")
                if not _is_int(accuracy_count):
                    cell_errors.append(f"{label}.accuracy_count must be an integer")
                if not _is_int(syntax_count):
                    cell_errors.append(f"{label}.syntax_count must be an integer")
                if _is_int(n) and n > 0:
                    if _is_int(accuracy_count) and not 0 <= accuracy_count <= n:
                        cell_errors.append(f"{label}.accuracy_count must be between 0 and n")
                    if _is_int(syntax_count) and not 0 <= syntax_count <= n:
                        cell_errors.append(f"{label}.syntax_count must be between 0 and n")
                    if not allow_cross_n and n != target_n:
                        cell_errors.append(
                            f"{label}.n={n} differs from target_n={target_n}; "
                            "set allow_cross_n with a reason to review this comparison"
                        )
                if (
                    isinstance(strategy, str)
                    and strategy.strip()
                    and _is_int(n)
                    and n > 0
                    and _is_int(accuracy_count)
                    and 0 <= accuracy_count <= n
                    and _is_int(syntax_count)
                    and 0 <= syntax_count <= n
                ):
                    cell_errors.extend(_source_display_errors(catalog_row, source, label))
                    validated_groups[group_name].append(source)

        if len(contexts) > 1:
            cell_errors.append(
                "source rows have different board/dataset/class heading contexts; "
                "the cohort is ambiguous"
            )

        baseline_rows = validated_groups["baseline"]
        metadecode_rows = validated_groups["metadecode"]
        baseline_maxima: dict[str, Any] = {}
        targets: dict[str, Any] = {"target_n": target_n}
        winning_refs: list[str] = []
        verdict = "invalid"
        if baseline_rows:
            max_accuracy = max(
                Fraction(row["accuracy_count"], row["n"]) for row in baseline_rows
            )
            max_syntax = max(
                Fraction(row["syntax_count"], row["n"]) for row in baseline_rows
            )
            syntax_ratio = min(max_syntax, Fraction(9, 10))
            min_accuracy_count = (max_accuracy.numerator * target_n) // max_accuracy.denominator + 1
            min_syntax_count = _ceil_fraction_times(syntax_ratio, target_n)
            baseline_maxima = {
                "accuracy": _fraction_summary(baseline_rows, "accuracy"),
                "syntax": _fraction_summary(baseline_rows, "syntax"),
            }
            targets.update(
                {
                    "min_accuracy_count": min_accuracy_count,
                    "syntax_ratio": str(syntax_ratio),
                    "min_syntax_count": min_syntax_count,
                }
            )
            winning_refs = [
                row["source_ref"]
                for row in metadecode_rows
                if Fraction(row["accuracy_count"], row["n"]) > max_accuracy
                and Fraction(row["syntax_count"], row["n"]) >= syntax_ratio
            ]
            if not cell_errors:
                if max_accuracy == 1:
                    verdict = "needs_review"
                    errors.append(
                        f"{cell_id}: strict accuracy target is impossible because "
                        "the maximum baseline accuracy is 1"
                    )
                elif winning_refs:
                    verdict = "current_win"
                else:
                    verdict = "candidate"
                    try:
                        recipe_json = _recipe_json(cell)
                    except (TypeError, ValueError, json.JSONDecodeError) as error:
                        cell_errors.append(str(error))
                        verdict = "invalid"
                    else:
                        candidates.append(
                            {
                                "cell_id": cell_id,
                                "dataset": cell.get("dataset", ""),
                                "model": cell.get("model", ""),
                                "class": cell.get("class", ""),
                                "target_n": target_n,
                                "min_accuracy_count": min_accuracy_count,
                                "min_syntax_count": min_syntax_count,
                                "recipe_json": recipe_json,
                            }
                        )

        if cell_errors:
            errors.extend(f"{cell_id}: {error}" for error in cell_errors)
            verdict = "invalid"
        cells_audit.append(
            {
                "cell_id": cell_id,
                **{field: cell.get(field) for field in COHORT_FIELDS},
                "target_n": target_n,
                "allow_cross_n": allow_cross_n,
                "cross_n_reason": cross_n_reason,
                "refs": {
                    "baseline": [row.get("source_ref") for row in cell.get("baseline_rows", []) if isinstance(row, dict)],
                    "metadecode": [row.get("source_ref") for row in cell.get("metadecode_rows", []) if isinstance(row, dict)],
                },
                "baseline_maxima": baseline_maxima,
                "targets": targets,
                "winning_metadecode_refs": winning_refs,
                "verdict": verdict,
                "errors": cell_errors,
            }
        )

    exclusions = reviewed.get("excluded", []) if isinstance(reviewed, dict) else []
    if not isinstance(exclusions, list):
        errors.append("reviewed excluded must be a list")
        exclusions = []
    for exclusion_index, exclusion in enumerate(exclusions):
        label = f"excluded[{exclusion_index}]"
        if not isinstance(exclusion, dict):
            errors.append(f"{label} must be an object")
            continue
        source_ref = exclusion.get("source_ref")
        reason = exclusion.get("reason")
        if not isinstance(source_ref, str) or source_ref not in catalog_by_ref:
            errors.append(f"{label} has unknown or stale source_ref {source_ref!r}")
            continue
        if source_ref in used_refs:
            errors.append(
                f"source_ref {source_ref} is used more than once "
                f"({used_refs[source_ref]} and {label})"
            )
            continue
        if not isinstance(reason, str) or not reason.strip():
            errors.append(f"{label}.reason must be a nonempty string")
            continue
        used_refs[source_ref] = label
        ref_groups["excluded"].append(source_ref)

    uncovered_refs = [row["source_ref"] for row in catalog if row["source_ref"] not in used_refs]
    if uncovered_refs:
        errors.append("catalog rows are not reviewed: " + ", ".join(uncovered_refs))

    if errors:
        candidates = []
    audit = {
        "matrix_sha256": matrix_sha256,
        "pre_rebar_snapshot_sha256": snapshot_sha256,
        "reviewed_matrix_sha256": reviewed_sha,
        "coverage": {
            "catalog_rows": len(catalog),
            "covered_rows": len(used_refs),
            "baseline_rows": len(ref_groups["baseline"]),
            "metadecode_rows": len(ref_groups["metadecode"]),
            "excluded_rows": len(ref_groups["excluded"]),
            "uncovered_refs": uncovered_refs,
        },
        "catalog": catalog,
        "refs": ref_groups,
        "cells": cells_audit,
        "candidate_manifest_sha256": None,
        "candidate_cells": [],
        "verdict": "error" if errors else "ok",
        "errors": errors,
    }
    return audit, candidates


def write_outputs(
    audit: dict[str, Any],
    candidates: list[dict[str, Any]],
    audit_path: Path,
    candidate_path: Path,
) -> None:
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    if audit["errors"]:
        candidate_path.unlink(missing_ok=True)
        audit_path.write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    with candidate_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANDIDATE_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(candidates)
    audit["candidate_manifest_sha256"] = sha256_file(candidate_path)
    audit["candidate_cells"] = [row["cell_id"] for row in candidates]
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--reviewed-json", type=Path, required=True)
    parser.add_argument("--snapshot-json", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--candidate-tsv", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    LOGGER.info(
        "[metadecode-rebar-scan] matrix=%s reviewed=%s",
        args.matrix,
        args.reviewed_json,
    )
    try:
        audit, candidates = scan_reviewed_matrix(
            args.matrix, args.reviewed_json, args.snapshot_json
        )
        write_outputs(audit, candidates, args.audit_json, args.candidate_tsv)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        args.candidate_tsv.unlink(missing_ok=True)
        LOGGER.error("[metadecode-rebar-scan] input/output error: %s", error)
        return 2
    if audit["errors"]:
        for error in audit["errors"]:
            LOGGER.error("[metadecode-rebar-scan] %s", error)
        return 2
    LOGGER.info(
        "[metadecode-rebar-scan] catalog_rows=%d cells=%d candidates=%d audit=%s manifest=%s",
        audit["coverage"]["catalog_rows"],
        len(audit["cells"]),
        len(candidates),
        args.audit_json,
        args.candidate_tsv,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
