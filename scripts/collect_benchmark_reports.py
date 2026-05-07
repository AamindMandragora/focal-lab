#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


SKIP_SUFFIXES = (
    "_cars_jobs.json",
    "_jobs.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy benchmark report JSONs from master experiment runs into one directory."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/generated-csd"),
        help="Generated CSD output directory containing master_experiments/.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        required=True,
        help="Directory where benchmark JSONs and indexes should be written.",
    )
    parser.add_argument(
        "--run-stamp",
        default=None,
        help="Only collect from master experiment directories containing this stamp.",
    )
    parser.add_argument(
        "--run-prefix",
        default="handoff_",
        help="Only collect from master experiment directories with this prefix.",
    )
    parser.add_argument(
        "--run-glob",
        default="*",
        help="Glob applied under master_experiments/ before run-prefix/run-stamp filters.",
    )
    parser.add_argument(
        "--include-helper-json",
        action="store_true",
        help="Also copy helper JSON files such as CARS jobs files.",
    )
    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="Do not remove previously copied report JSONs from the destination first.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:
        return {"_read_error": str(exc)}
    return payload if isinstance(payload, dict) else {"_payload_type": type(payload).__name__}


def nested_get(payload: dict[str, Any], keys: tuple[str, ...]) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def first_present(payload: dict[str, Any], paths: tuple[tuple[str, ...], ...]) -> Any:
    for path in paths:
        value = nested_get(payload, path)
        if value is not None:
            return value
    return None


def infer_name_fields(report_path: Path) -> dict[str, str | None]:
    stem = report_path.stem
    parts = stem.split("_")
    dataset = parts[0] if parts else None
    method = parts[1] if len(parts) > 1 else None
    model = "_".join(parts[2:]) if len(parts) > 2 else None
    return {
        "benchmark_name": stem,
        "dataset_from_filename": dataset,
        "method_from_filename": method,
        "model_from_filename": model,
    }


def is_helper_json(path: Path) -> bool:
    return path.name.endswith(SKIP_SUFFIXES)


def discover_reports(args: argparse.Namespace) -> list[Path]:
    master_dir = args.output_dir / "master_experiments"
    if not master_dir.exists():
        return []

    reports: list[Path] = []
    for run_dir in sorted(master_dir.glob(args.run_glob)):
        if not run_dir.is_dir():
            continue
        if args.run_prefix and not run_dir.name.startswith(args.run_prefix):
            continue
        if args.run_stamp and args.run_stamp not in run_dir.name:
            continue
        benchmark_dir = run_dir / "benchmarks"
        if not benchmark_dir.is_dir():
            continue
        for report in sorted(benchmark_dir.glob("*.json")):
            if not args.include_helper_json and is_helper_json(report):
                continue
            reports.append(report)
    return reports


def make_record(source: Path, copied: Path, payload: dict[str, Any]) -> dict[str, Any]:
    run_name = source.parents[1].name
    inferred = infer_name_fields(source)
    config = payload.get("config") if isinstance(payload.get("config"), dict) else {}

    return {
        "run_name": run_name,
        "benchmark_name": inferred["benchmark_name"],
        "dataset": config.get("dataset") or inferred["dataset_from_filename"],
        "method": config.get("method") or inferred["method_from_filename"],
        "model": (
            config.get("eval_model")
            or config.get("model")
            or config.get("model_name")
            or inferred["model_from_filename"]
        ),
        "accuracy": first_present(payload, (("accuracy",), ("all_exec_accuracy",))),
        "syntax_rate": first_present(payload, (("syntax_rate",), ("format_rate",))),
        "format_rate": payload.get("format_rate"),
        "num_correct": payload.get("num_correct"),
        "num_examples": payload.get("num_examples"),
        "accuracy_denominator": payload.get("accuracy_denominator"),
        "success": payload.get("success"),
        "error": payload.get("error"),
        "source_path": str(source.resolve()),
        "copied_path": str(copied.resolve()),
    }


def write_tsv(path: Path, records: list[dict[str, Any]]) -> None:
    columns = [
        "run_name",
        "benchmark_name",
        "dataset",
        "method",
        "model",
        "accuracy",
        "syntax_rate",
        "format_rate",
        "num_correct",
        "num_examples",
        "accuracy_denominator",
        "success",
        "error",
        "copied_path",
        "source_path",
    ]
    lines = ["\t".join(columns)]
    for record in records:
        row = []
        for column in columns:
            value = record.get(column)
            text = "" if value is None else str(value)
            row.append(text.replace("\t", " ").replace("\n", " "))
        lines.append("\t".join(row))
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.dest.mkdir(parents=True, exist_ok=True)
    if not args.no_clear:
        for old_report in args.dest.glob("*.json"):
            old_report.unlink()

    records: list[dict[str, Any]] = []
    for source in discover_reports(args):
        run_name = source.parents[1].name
        copied = args.dest / f"{run_name}__{source.name}"
        shutil.copy2(source, copied)
        payload = read_json(source)
        records.append(make_record(source, copied, payload))

    records.sort(key=lambda item: (str(item["run_name"]), str(item["benchmark_name"])))

    index = {
        "output_dir": str(args.output_dir),
        "dest": str(args.dest.resolve()),
        "run_prefix": args.run_prefix,
        "run_stamp": args.run_stamp,
        "num_reports": len(records),
        "reports": records,
    }
    (args.dest / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    write_tsv(args.dest / "index.tsv", records)
    (args.dest / "README.txt").write_text(
        "Consolidated benchmark reports.\n"
        "\n"
        "Each copied JSON is named <master-run-directory>__<original-report-name>.json\n"
        "so repeated benchmark names from different phases/models do not collide.\n"
        "index.tsv is the quick scan view; index.json preserves the same metadata as JSON.\n"
    )

    print(f"[collect] copied {len(records)} benchmark report(s) to {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
