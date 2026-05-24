#!/usr/bin/env python3
"""
Split combined SMILES baseline JSONs (all classes in one file) into per-class files.

Some in-repo adapters historically loaded every SMILES class while writing
``smiles__class_<name>__...`` paths. Row-level ``correct`` / ``syntax_valid`` are
already per-class; this script filters ``answers`` by ``question`` (class name)
and recomputes aggregate accuracy, syntax_rate, and timing metrics.

Usage (repo root):
  python -m synthesis.scripts.split_smiles_class_baselines --dry-run
  python -m synthesis.scripts.split_smiles_class_baselines --apply --backup
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from synthesis.evaluate.baseline_store import build_minimal_baseline_from_rows
from synthesis.evaluate.benchmarks.smiles.dataset import SMILES_CLASSES

AFFECTED_STRATEGIES = ("gcd", "itergen", "cars", "rejection_sampling")
CLASS_IN_NAME = re.compile(r"__class_(?P<class>[^_]+)__")


def _class_from_path(path: Path) -> str | None:
    match = CLASS_IN_NAME.search(path.name)
    return match.group("class") if match else None


def _group_key(path: Path) -> str:
    """Paths that differ only by class name share one group."""
    cls = _class_from_path(path)
    if not cls:
        return str(path)
    return str(path).replace(f"__class_{cls}__", "__class_*__")


def _answers_by_class(answers: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {name: [] for name in SMILES_CLASSES}
    for row in answers:
        key = str(row.get("question") or row.get("class_name") or "").strip()
        if key in grouped:
            grouped[key].append(row)
    return grouped


def _needs_split(path: Path, payload: dict[str, Any]) -> bool:
    expected = _class_from_path(path)
    if expected is None:
        return False
    answers = payload.get("answers") or []
    if not answers:
        return False
    classes_present = {str(a.get("question") or "").strip() for a in answers}
    if len(classes_present) > 1:
        return True
    if len(answers) != 100:
        return True
    if classes_present != {expected}:
        return True
    return False


def _prorate_wall_time(
    original: float | None, *, kept: int, total: int
) -> float | None:
    if original is None or total <= 0:
        return original
    return round(float(original) * (kept / total), 4)


def _split_payload(
    source: dict[str, Any],
    *,
    class_name: str,
    rows: list[dict[str, Any]],
    total_source_rows: int,
) -> dict[str, Any]:
    metadata = dict(source.get("metadata") or {})
    metadata.pop("checkpoint", None)
    metadata["complete"] = True
    metadata["smiles_class_split_from_combined"] = True
    metadata["smiles_class"] = class_name

    extra_metrics = dict(source.get("metrics") or {})
    for key in (
        "num_examples",
        "total_generation_seconds",
        "mean_generation_seconds_per_example",
        "examples_with_generation_timing",
        "total_output_tokens",
        "mean_output_tokens_per_example",
        "examples_with_token_counts",
        "checkpoint_examples",
    ):
        extra_metrics.pop(key, None)
    adapter = extra_metrics.pop("adapter", None)

    wall = source.get("metrics", {}).get("run_wall_time_seconds")
    if wall is None:
        wall = metadata.get("run_wall_time_seconds")

    payload = build_minimal_baseline_from_rows(
        rows,
        dataset="smiles",
        run_wall_time_seconds=_prorate_wall_time(
            wall, kept=len(rows), total=total_source_rows
        ),
        extra_metrics={"adapter": adapter} if adapter else None,
        metadata=metadata,
    )
    return payload


def _pick_canonical_source(paths: list[Path]) -> tuple[Path, dict[str, Any]]:
    best_path = paths[0]
    best_payload: dict[str, Any] = {}
    best_len = -1
    for path in paths:
        payload = json.loads(path.read_text())
        n = len(payload.get("answers") or [])
        if n > best_len:
            best_len = n
            best_path = path
            best_payload = payload
    return best_path, best_payload


def split_baselines(
    *,
    baselines_root: Path,
    strategies: tuple[str, ...],
    samples_per_class: int,
    dry_run: bool,
    backup: bool,
) -> int:
    groups: dict[str, list[Path]] = {}
    for strategy in strategies:
        strategy_dir = baselines_root / strategy
        if not strategy_dir.is_dir():
            continue
        for path in sorted(strategy_dir.rglob("smiles__class_*.json")):
            groups.setdefault(_group_key(path), []).append(path)

    actions = 0
    for group_paths in groups.values():
        canonical_path, payload = _pick_canonical_source(group_paths)
        if not any(_needs_split(p, payload) for p in group_paths):
            continue

        answers = payload.get("answers") or []
        by_class = _answers_by_class(answers)
        total = len(answers)

        for class_name in SMILES_CLASSES:
            rows = by_class[class_name]
            if not rows:
                print(
                    f"[skip] {canonical_path.parent.name}: no rows for {class_name} "
                    f"(source had {dict(Counter(a.get('question') for a in answers))})"
                )
                continue

            out_path = next(
                (p for p in group_paths if _class_from_path(p) == class_name),
                None,
            )
            if out_path is None:
                cls = _class_from_path(canonical_path)
                if not cls:
                    continue
                out_path = Path(
                    str(canonical_path).replace(f"__class_{cls}__", f"__class_{class_name}__")
                )

            status = "ok" if len(rows) == samples_per_class else f"partial({len(rows)}/{samples_per_class})"
            print(
                f"[{'dry-run' if dry_run else 'write'}] {out_path.name}: "
                f"{len(rows)} {class_name} rows ({status}) from {canonical_path.name} ({total} total)"
            )

            if dry_run:
                actions += 1
                continue

            if backup and out_path.is_file():
                bak = out_path.with_suffix(out_path.suffix + ".bak")
                if not bak.is_file():
                    shutil.copy2(out_path, bak)

            split_payload = _split_payload(
                payload,
                class_name=class_name,
                rows=rows,
                total_source_rows=total,
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(split_payload, indent=2) + "\n")
            actions += 1

    return actions


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baselines-root",
        type=Path,
        default=repo_root / "outputs" / "baselines",
    )
    parser.add_argument(
        "--strategies",
        default=",".join(AFFECTED_STRATEGIES),
        help=f"Comma-separated strategies (default: {','.join(AFFECTED_STRATEGIES)})",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=100,
        help="Expected rows per class after split (default: 100)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print actions only")
    parser.add_argument("--apply", action="store_true", help="Write split JSON files")
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Keep a .json.bak copy before overwriting an existing target file",
    )
    args = parser.parse_args()

    if not args.dry_run and not args.apply:
        parser.error("Specify --dry-run or --apply")

    strategies = tuple(s.strip() for s in args.strategies.split(",") if s.strip())
    n = split_baselines(
        baselines_root=args.baselines_root.expanduser(),
        strategies=strategies,
        samples_per_class=args.samples_per_class,
        dry_run=args.dry_run,
        backup=args.backup,
    )
    print(f"Done: {n} file(s) {'would be ' if args.dry_run else ''}written.")


if __name__ == "__main__":
    main()
