#!/usr/bin/env python3
"""
Collect baseline and synthesis results into paper-ready LaTeX table fragments.

Reads baseline JSONs from outputs/baselines/ and metadecode success reports
from outputs/generated/, then emits:
  1. Main results table (Table 1)
  2. Step-budget ablation (Table 2)
  3. Synthesis-iterations ablation (Table 3)
  4. Synthesizer-model ablation (Table 4)
  5. Beam/mask/policy factorial ablation (Table 5)

Usage:
  python -m synthesis.scripts.collect_paper_results
  python -m synthesis.scripts.collect_paper_results --baselines-dir outputs/baselines --generated-dir outputs/generated
  python -m synthesis.scripts.collect_paper_results --paper-main-table --paper-bold-best
    # paste Table 1 tabular rows into paper/experiments.tex
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _json_relpath_under_repo(path: Path, repo_root: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def _git_tracked_json_relpaths(repo_root: Path) -> frozenset[str]:
    """Paths under outputs/ that are tracked by git and end in .json (posix relpaths)."""
    proc = subprocess.run(
        ["git", "-C", str(repo_root), "ls-files", "-z", "--", "outputs"],
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return frozenset()
    out: list[str] = []
    for chunk in proc.stdout.split(b"\0"):
        if not chunk:
            continue
        s = chunk.decode(errors="replace").replace("\\", "/")
        if s.endswith(".json"):
            out.append(s)
    return frozenset(out)


MODELS = [
    ("Qwen/Qwen2.5-Coder-1.5B-Instruct", "1.5B"),
    ("Qwen/Qwen2.5-Coder-7B-Instruct", "7B"),
    ("Qwen/Qwen2.5-Coder-14B-Instruct", "14B"),
    ("meta-llama/Llama-3.1-8B-Instruct", "Llama-8B"),
]
# First-column \\multirow labels for paper/experiments.tex Table~\\ref{tab:main_results}.
MODEL_TABULAR_MACROS = [
    r"\QwenSmall",
    r"\QwenCoder",
    r"\QwenBig",
    r"\LlamaEight",
]
BENCHMARKS = ["gsm_symbolic", "spider", "smiles"]
STRATEGIES = ["unconstrained", "gcd", "crane", "itergen", "cars", "metadecode"]
SMILES_CLASSES = ("acrylates", "chain_extenders", "isocyanates")
MASK_LABELS = ("mask_on", "mask_off")


def _slugify(s: str) -> str:
    return s.replace("/", "_").replace(":", "_").replace(" ", "_").replace("-", "_")


def _record_num_examples(record: dict[str, Any]) -> int:
    metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
    for source in (record, metrics):
        for key in ("num_examples", "sample_count"):
            value = source.get(key)
            if isinstance(value, int) and value > 0:
                return value
            if isinstance(value, float) and value > 0:
                return int(value)
    for key in ("answers", "sample_outputs"):
        value = record.get(key)
        if isinstance(value, list) and value:
            return len(value)
    return 1


def _aggregate_records(records: list[dict[str, Any] | None]) -> dict[str, Any] | None:
    present = [record for record in records if record is not None]
    if not present:
        return None
    total = sum(_record_num_examples(record) for record in present)
    if total <= 0:
        total = len(present)
    accuracy = sum(
        float(record.get("accuracy", 0.0)) * _record_num_examples(record)
        for record in present
    ) / total
    syntax_rate = sum(
        float(record.get("syntax_rate", 0.0)) * _record_num_examples(record)
        for record in present
    ) / total
    return {
        "accuracy": accuracy,
        "syntax_rate": syntax_rate,
        "metrics": {
            "num_examples": total,
            "num_records": len(present),
        },
    }


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _matches_name_filters(path: Path, **kwargs: Any) -> bool:
    name = path.name
    for key, value in kwargs.items():
        if value is None:
            continue
        if key == "max_steps" and f"__ms{value}" not in name:
            return False
        if key == "token_budget" and f"__tb{value}" not in name:
            return False
    return True


def _load_baseline_single(
    baselines_dir: Path,
    strategy: str,
    model: str,
    benchmark_key: str,
    *,
    repo_root: Path | None = None,
    tracked_relpaths: frozenset[str] | None = None,
    **kwargs: Any,
) -> dict[str, Any] | None:
    model_slug = _slugify(model)
    model_dir = baselines_dir / strategy / model_slug
    candidates = []
    if model_dir.is_dir():
        candidates = sorted(
            (
                path
                for path in model_dir.glob(f"{benchmark_key}__*.json")
                if _matches_name_filters(path, **kwargs)
            ),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    if tracked_relpaths is not None and repo_root is not None:
        candidates = [
            p
            for p in candidates
            if _json_relpath_under_repo(p, repo_root) in tracked_relpaths
        ]
    if not candidates:
        return None
    return _load_json(candidates[0])


def _load_baseline(
    baselines_dir: Path,
    strategy: str,
    model: str,
    benchmark: str,
    *,
    smiles_classes: tuple[str, ...] = SMILES_CLASSES,
    repo_root: Path | None = None,
    tracked_relpaths: frozenset[str] | None = None,
    **kwargs: Any,
) -> dict[str, Any] | None:
    if benchmark == "smiles":
        class_records = [
            _load_baseline_single(
                baselines_dir,
                strategy,
                model,
                f"smiles__class_{_slugify(class_name)}",
                repo_root=repo_root,
                tracked_relpaths=tracked_relpaths,
                **kwargs,
            )
            for class_name in smiles_classes
        ]
        if any(record is not None for record in class_records):
            if any(record is None for record in class_records):
                return None
            return _aggregate_records(class_records)

    return _load_baseline_single(
        baselines_dir,
        strategy,
        model,
        benchmark,
        repo_root=repo_root,
        tracked_relpaths=tracked_relpaths,
        **kwargs,
    )


def _payload_from_success_report(report: dict[str, Any]) -> dict[str, Any]:
    ev = report.get("evaluation_result") or {}
    samples = report.get("sample_outputs") or []
    return {
        "accuracy": ev.get("accuracy", 0.0),
        "syntax_rate": ev.get("syntax_rate", 0.0),
        "metrics": {"num_examples": len(samples) or 1},
    }


def _load_generated_prefix(
    generated_dir: Path,
    prefix: str,
    *,
    repo_root: Path | None = None,
    tracked_relpaths: frozenset[str] | None = None,
) -> dict[str, Any] | None:
    best: Path | None = None
    best_mtime = -1.0

    if not generated_dir.is_dir():
        return None

    for run_dir in generated_dir.iterdir():
        if not run_dir.is_dir() or not run_dir.name.startswith(prefix):
            continue
        success = run_dir / "results" / "success_report.json"
        if not success.is_file():
            continue
        if tracked_relpaths is not None and repo_root is not None:
            rel = _json_relpath_under_repo(success, repo_root)
            if rel not in tracked_relpaths:
                continue
        mtime = success.stat().st_mtime
        if mtime > best_mtime:
            best = success
            best_mtime = mtime

    if best is None:
        return None
    report = _load_json(best)
    if report is None:
        return None
    return _payload_from_success_report(report)


def _generated_run_matches(name: str, **kwargs: Any) -> bool:
    for key, value in kwargs.items():
        if value is None:
            continue
        if key == "synth_iter":
            if f"_iter{value}_" not in name and not name.endswith(f"_iter{value}"):
                return False
        elif key == "gen_profile":
            if _slugify(str(value)) not in name:
                return False
        elif key == "max_steps":
            if f"_ms{value}_" not in name and not name.endswith(f"_ms{value}"):
                return False
    return True


def _load_metadecode(
    generated_dir: Path,
    benchmark: str,
    model: str,
    *,
    smiles_classes: tuple[str, ...] = SMILES_CLASSES,
    smiles_class: str | None = None,
    baselines_dir: Path | None = None,
    repo_root: Path | None = None,
    tracked_relpaths: frozenset[str] | None = None,
    **kwargs: Any,
) -> dict[str, Any] | None:
    """Find the most recent metadecode success report matching the given criteria."""
    if benchmark == "smiles" and smiles_class is None:
        class_records = [
            _load_metadecode(
                generated_dir,
                benchmark,
                model,
                smiles_classes=smiles_classes,
                smiles_class=class_name,
                baselines_dir=baselines_dir,
                repo_root=repo_root,
                tracked_relpaths=tracked_relpaths,
                **kwargs,
            )
            for class_name in smiles_classes
        ]
        if any(record is not None for record in class_records):
            if any(record is None for record in class_records):
                return None
            return _aggregate_records(class_records)

    model_slug = _slugify(model)
    prefix = f"metadecode_{benchmark}_{model_slug}"
    class_token = f"_class_{_slugify(smiles_class)}" if smiles_class else None
    best: Path | None = None
    best_mtime = -1.0

    if generated_dir.is_dir():
        for run_dir in generated_dir.iterdir():
            if not run_dir.is_dir() or not run_dir.name.startswith(prefix):
                continue
            if class_token is not None and class_token not in run_dir.name:
                continue
            success = run_dir / "results" / "success_report.json"
            if not success.is_file():
                continue
            if not _generated_run_matches(run_dir.name, **kwargs):
                continue
            if tracked_relpaths is not None and repo_root is not None:
                rel = _json_relpath_under_repo(success, repo_root)
                if rel not in tracked_relpaths:
                    continue

            mtime = success.stat().st_mtime
            if mtime > best_mtime:
                best = success
                best_mtime = mtime

    if best is not None:
        report = _load_json(best)
        if report is None:
            return None
        return _payload_from_success_report(report)

    if baselines_dir is not None:
        return _load_baseline_single(
            baselines_dir,
            "metadecode",
            model,
            benchmark,
            repo_root=repo_root,
            tracked_relpaths=tracked_relpaths,
            **kwargs,
        )
    return None


def _load_factorial_run(
    generated_dir: Path,
    benchmark: str,
    beam_size: int,
    mask_label: str,
    policy: str,
    *,
    smiles_classes: tuple[str, ...] = SMILES_CLASSES,
    smiles_class: str | None = None,
    repo_root: Path | None = None,
    tracked_relpaths: frozenset[str] | None = None,
) -> dict[str, Any] | None:
    if benchmark == "smiles" and smiles_class is None:
        class_records = [
            _load_factorial_run(
                generated_dir,
                benchmark,
                beam_size,
                mask_label,
                policy,
                smiles_classes=smiles_classes,
                smiles_class=class_name,
                repo_root=repo_root,
                tracked_relpaths=tracked_relpaths,
            )
            for class_name in smiles_classes
        ]
        if any(record is not None for record in class_records):
            if any(record is None for record in class_records):
                return None
            return _aggregate_records(class_records)

    class_suffix = ""
    if benchmark == "smiles":
        if smiles_class is None:
            return None
        class_suffix = f"_class_{_slugify(smiles_class)}"
    prefix = f"ablat_beam{beam_size}_{mask_label}_{policy}_{benchmark}{class_suffix}"
    return _load_generated_prefix(
        generated_dir,
        prefix,
        repo_root=repo_root,
        tracked_relpaths=tracked_relpaths,
    )


def _fmt(value: float | None, pct: bool = True, *, bold: bool = False) -> str:
    if value is None:
        return r"\todo{--}"
    v = value * 100 if pct else value
    s = f"{v:.0f}"
    return f"\\textbf{{{s}}}" if bold else s


def _strategy_display(strategy: str) -> str:
    return {
        "unconstrained": "Unconstr.",
        "gcd": r"\GCD",
        "crane": r"\Crane",
        "itergen": r"\IterGen",
        "cars": r"\CARS",
        "metadecode": r"\Tool",
    }.get(strategy, strategy)


def _main_table_rows_raw(
    baselines_dir: Path,
    generated_dir: Path,
    *,
    smiles_classes: tuple[str, ...] = SMILES_CLASSES,
    repo_root: Path | None = None,
    tracked_relpaths: frozenset[str] | None = None,
) -> list[tuple[str, list[tuple[str, list[tuple[float | None, float | None]]]]]]:
    """Per model: list of (strategy, [(acc, cw) per benchmark])."""
    blocks: list[tuple[str, list[tuple[str, list[tuple[float | None, float | None]]]]]] = []
    for model_full, _model_short in MODELS:
        rows: list[tuple[str, list[tuple[float | None, float | None]]]] = []
        for strategy in STRATEGIES:
            pairs: list[tuple[float | None, float | None]] = []
            for benchmark in BENCHMARKS:
                if strategy == "metadecode":
                    data = _load_metadecode(
                        generated_dir,
                        benchmark,
                        model_full,
                        smiles_classes=smiles_classes,
                        baselines_dir=baselines_dir,
                        repo_root=repo_root,
                        tracked_relpaths=tracked_relpaths,
                    )
                else:
                    data = _load_baseline(
                        baselines_dir,
                        strategy,
                        model_full,
                        benchmark,
                        smiles_classes=smiles_classes,
                        repo_root=repo_root,
                        tracked_relpaths=tracked_relpaths,
                    )
                if data is None:
                    pairs.append((None, None))
                else:
                    acc = data.get("accuracy")
                    syn = data.get("syntax_rate")
                    pairs.append(
                        (
                            float(acc) if isinstance(acc, (int, float)) else None,
                            float(syn) if isinstance(syn, (int, float)) else None,
                        )
                    )
            rows.append((strategy, pairs))
        blocks.append((model_full, rows))
    return blocks


def _flatten_pairs(pairs: list[tuple[float | None, float | None]]) -> list[float | None]:
    out: list[float | None] = []
    for acc, cw in pairs:
        out.append(acc)
        out.append(cw)
    return out


def emit_main_table(
    baselines_dir: Path,
    generated_dir: Path,
    *,
    smiles_classes: tuple[str, ...] = SMILES_CLASSES,
    paper_multirow: bool = False,
    bold_best: bool = False,
    repo_root: Path | None = None,
    tracked_relpaths: frozenset[str] | None = None,
) -> str:
    lines: list[str] = []
    raw_blocks = _main_table_rows_raw(
        baselines_dir,
        generated_dir,
        smiles_classes=smiles_classes,
        repo_root=repo_root,
        tracked_relpaths=tracked_relpaths,
    )

    for block_idx, (_model_full, rows) in enumerate(raw_blocks):
        flat = [_flatten_pairs(pairs) for _s, pairs in rows]
        bold_cells: set[tuple[int, int]] = set()
        if bold_best:
            n_metrics = len(BENCHMARKS) * 2
            for metric_col in range(n_metrics):
                vals = [
                    flat[r][metric_col]
                    for r in range(len(rows))
                    if flat[r][metric_col] is not None
                ]
                if not vals:
                    continue
                best = max(vals)
                for r in range(len(rows)):
                    if flat[r][metric_col] is not None and flat[r][metric_col] == best:
                        bold_cells.add((r, metric_col))

        model_macro = MODEL_TABULAR_MACROS[block_idx]
        for row_idx, (strategy, pairs) in enumerate(rows):
            row_cells: list[str] = []
            col = 0
            for benchmark_i, _benchmark in enumerate(BENCHMARKS):
                acc, syn = pairs[benchmark_i]
                row_cells.append(
                    _fmt(acc, bold=((row_idx, col) in bold_cells)),
                )
                col += 1
                row_cells.append(_fmt(syn, bold=((row_idx, col) in bold_cells)))
                col += 1

            strategy_label = _strategy_display(strategy)
            cells = " & ".join(row_cells)
            if paper_multirow:
                first_col = (
                    f"\\multirow{{6}}{{1.05cm}}{{{model_macro}}}"
                    if row_idx == 0
                    else ""
                )
                lines.append(f"{first_col} & {strategy_label:<12s} & {cells} \\\\")
            else:
                lines.append(f"& {strategy_label:<12s} & {cells} \\\\")
        lines.append(r"\midrule")

    if lines and lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"

    return "\n".join(lines)


def emit_step_budget_table(
    baselines_dir: Path,
    generated_dir: Path,
    step_budgets: list[int],
    *,
    smiles_classes: tuple[str, ...] = SMILES_CLASSES,
    repo_root: Path | None = None,
    tracked_relpaths: frozenset[str] | None = None,
) -> str:
    lines: list[str] = []
    ablation_model = "Qwen/Qwen2.5-Coder-7B-Instruct"

    for strategy in ["gcd", "crane", "itergen", "cars", "metadecode"]:
        strategy_label = {
            "gcd": r"\GCD",
            "crane": r"\Crane",
            "itergen": r"\IterGen",
            "cars": r"\CARS",
            "metadecode": r"\Tool",
        }.get(strategy, strategy)

        gsm_cells: list[str] = []
        spider_cells: list[str] = []
        smiles_cells: list[str] = []
        for ms in step_budgets:
            for benchmark, cells in [("gsm_symbolic", gsm_cells), ("spider", spider_cells), ("smiles", smiles_cells)]:
                if strategy == "metadecode":
                    data = _load_metadecode(
                        generated_dir,
                        benchmark,
                        ablation_model,
                        max_steps=ms,
                        smiles_classes=smiles_classes,
                        baselines_dir=baselines_dir,
                        repo_root=repo_root,
                        tracked_relpaths=tracked_relpaths,
                    )
                else:
                    data = _load_baseline(
                        baselines_dir,
                        strategy,
                        ablation_model,
                        benchmark,
                        max_steps=ms,
                        smiles_classes=smiles_classes,
                        repo_root=repo_root,
                        tracked_relpaths=tracked_relpaths,
                    )
                cells.append(_fmt(data.get("accuracy") if data else None))

        all_cells = " & ".join(gsm_cells + spider_cells + smiles_cells)
        lines.append(f"{strategy_label:<12s} & {all_cells} \\\\")

    return "\n".join(lines)


def emit_synth_iter_table(
    generated_dir: Path,
    synth_iters: list[int],
    *,
    baselines_dir: Path | None = None,
    smiles_classes: tuple[str, ...] = SMILES_CLASSES,
    repo_root: Path | None = None,
    tracked_relpaths: frozenset[str] | None = None,
) -> str:
    lines_verif: list[str] = []
    lines_acc: list[str] = []
    ablation_model = "Qwen/Qwen2.5-Coder-7B-Instruct"

    for benchmark in ["gsm_symbolic", "spider", "smiles"]:
        verif_cells: list[str] = []
        acc_cells: list[str] = []
        for k in synth_iters:
            data = _load_metadecode(
                generated_dir,
                benchmark,
                ablation_model,
                synth_iter=k,
                smiles_classes=smiles_classes,
                baselines_dir=baselines_dir,
                repo_root=repo_root,
                tracked_relpaths=tracked_relpaths,
            )
            acc_cells.append(_fmt(data.get("accuracy") if data else None))
            verif_cells.append(r"\todo{--}")
        lines_verif.extend(verif_cells)
        lines_acc.extend(acc_cells)

    verif_row = r"Verif.\ pass rate (\%) & " + " & ".join(lines_verif) + r" \\"
    acc_row = r"Accuracy (\%)          & " + " & ".join(lines_acc) + r" \\"
    return verif_row + "\n" + acc_row


def emit_synth_model_table(
    generated_dir: Path,
    gen_profiles: list[str],
    *,
    baselines_dir: Path | None = None,
    smiles_classes: tuple[str, ...] = SMILES_CLASSES,
    repo_root: Path | None = None,
    tracked_relpaths: frozenset[str] | None = None,
) -> str:
    lines: list[str] = []
    ablation_model = "Qwen/Qwen2.5-Coder-7B-Instruct"
    labels = {
        "gpt5.4": r"\SynthGPT",
        "opus4.7": r"\SynthOpus",
        "gemini-pro": r"\SynthGemini",
    }

    for profile in gen_profiles:
        cells: list[str] = []
        for benchmark in ["gsm_symbolic", "spider", "smiles"]:
            data = _load_metadecode(
                generated_dir,
                benchmark,
                ablation_model,
                gen_profile=profile,
                smiles_classes=smiles_classes,
                baselines_dir=baselines_dir,
                repo_root=repo_root,
                tracked_relpaths=tracked_relpaths,
            )
            cells.append(r"\todo{--}")  # verif pass rate
            cells.append(_fmt(data.get("accuracy") if data else None))
        label = labels.get(profile, profile)
        all_cells = " & ".join(cells)
        lines.append(f"{label:<16s} & {all_cells} \\\\")

    return "\n".join(lines)


def emit_factorial_table(
    generated_dir: Path,
    beam_sizes: list[int],
    helper_policies: list[str],
    *,
    smiles_classes: tuple[str, ...] = SMILES_CLASSES,
    repo_root: Path | None = None,
    tracked_relpaths: frozenset[str] | None = None,
) -> str:
    lines: list[str] = []
    for beam_size in beam_sizes:
        for mask_label in MASK_LABELS:
            for policy in helper_policies:
                cells: list[str] = []
                for benchmark in BENCHMARKS:
                    data = _load_factorial_run(
                        generated_dir,
                        benchmark,
                        beam_size,
                        mask_label,
                        policy,
                        smiles_classes=smiles_classes,
                        repo_root=repo_root,
                        tracked_relpaths=tracked_relpaths,
                    )
                    cells.append(_fmt(data.get("accuracy") if data else None))
                mask_display = mask_label.replace("_", r"\_")
                lines.append(
                    f"{beam_size} & {mask_display:<9s} & {policy:<7s} & "
                    + " & ".join(cells)
                    + r" \\"
                )
    return "\n".join(lines)


def main() -> None:
    repo = _repo_root()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baselines-dir", type=Path, default=repo / "outputs" / "baselines")
    p.add_argument("--generated-dir", type=Path, default=repo / "outputs" / "generated")
    p.add_argument("--step-budgets", default="256,512,1024")
    p.add_argument("--synth-iters", default="3,5,10")
    p.add_argument("--gen-profiles", default="gpt5.4,opus4.7,gemini-pro")
    p.add_argument("--smiles-classes", default=",".join(SMILES_CLASSES))
    p.add_argument(
        "--paper-main-table",
        action="store_true",
        help="Emit Table 1 body with \\multirow model column for paper/experiments.tex",
    )
    p.add_argument(
        "--paper-bold-best",
        action="store_true",
        help="With --paper-main-table, bold best value per metric column within each model block",
    )
    p.add_argument("--beam-sizes", default="1,2,4")
    p.add_argument("--helper-policies", default="utility,bandit")
    p.add_argument(
        "--git-tracked-only",
        action="store_true",
        help="Only read JSON files tracked by git under outputs/ (via git ls-files)",
    )
    args = p.parse_args()

    step_budgets = [int(x) for x in args.step_budgets.split(",")]
    synth_iters = [int(x) for x in args.synth_iters.split(",")]
    gen_profiles = [x.strip() for x in args.gen_profiles.split(",")]
    smiles_classes = tuple(x.strip() for x in args.smiles_classes.split(",") if x.strip())
    beam_sizes = [int(x) for x in args.beam_sizes.split(",")]
    helper_policies = [x.strip() for x in args.helper_policies.split(",") if x.strip()]
    unknown_smiles = sorted(set(smiles_classes) - set(SMILES_CLASSES))
    if unknown_smiles:
        raise SystemExit(
            f"Unknown SMILES class(es): {unknown_smiles}. Expected one of {SMILES_CLASSES}."
        )

    tracked = _git_tracked_json_relpaths(repo) if args.git_tracked_only else None
    repo_for_track = repo if tracked is not None else None

    print("=" * 60)
    print("Table 1: Main Results")
    print("=" * 60)
    print(
        emit_main_table(
            args.baselines_dir,
            args.generated_dir,
            smiles_classes=smiles_classes,
            paper_multirow=args.paper_main_table,
            bold_best=args.paper_bold_best,
            repo_root=repo_for_track,
            tracked_relpaths=tracked,
        )
    )
    print()

    print("=" * 60)
    print("Table 2: Step Budget Ablation")
    print("=" * 60)
    print(
        emit_step_budget_table(
            args.baselines_dir,
            args.generated_dir,
            step_budgets,
            smiles_classes=smiles_classes,
            repo_root=repo_for_track,
            tracked_relpaths=tracked,
        )
    )
    print()

    print("=" * 60)
    print("Table 3: Synthesis Iterations Ablation")
    print("=" * 60)
    print(
        emit_synth_iter_table(
            args.generated_dir,
            synth_iters,
            baselines_dir=args.baselines_dir,
            smiles_classes=smiles_classes,
            repo_root=repo_for_track,
            tracked_relpaths=tracked,
        )
    )
    print()

    print("=" * 60)
    print("Table 4: Synthesizer Model Ablation")
    print("=" * 60)
    print(
        emit_synth_model_table(
            args.generated_dir,
            gen_profiles,
            baselines_dir=args.baselines_dir,
            smiles_classes=smiles_classes,
            repo_root=repo_for_track,
            tracked_relpaths=tracked,
        )
    )
    print()

    print("=" * 60)
    print("Table 5: Beam / Mask / Helper Policy Factorial")
    print("=" * 60)
    print(
        emit_factorial_table(
            args.generated_dir,
            beam_sizes,
            helper_policies,
            smiles_classes=smiles_classes,
            repo_root=repo_for_track,
            tracked_relpaths=tracked,
        )
    )


if __name__ == "__main__":
    main()
