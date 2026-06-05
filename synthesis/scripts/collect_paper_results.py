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

from synthesis.project_paths import (
    iter_generated_run_dirs,
    resolve_baseline_json_path,
    slugify as path_slugify,
)


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
BENCHMARKS = ["gsm_symbolic", "spider"]
STRATEGIES = [
    "unconstrained",
    "gcd",
    "crane",
    "itergen",
    "rs",
    "metadecode",
]
# Main matrix MetaDecode runs (run_all_tests.py --main-generation-model).
DEFAULT_MAIN_GEN_PROFILE = "gemini"
DEFAULT_ABLATION_GEN_PROFILES = ("sonnet4.6", "gpt5.5")
MASK_LABELS = ("mask_on", "mask_off")


def _slugify(s: str) -> str:
    return path_slugify(s)


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
    benchmark = benchmark_key
    smiles_class = ""
    if benchmark_key.startswith("smiles__class_"):
        benchmark = "smiles"
        smiles_class = benchmark_key.split("smiles__class_", 1)[1]

    token_budget = str(kwargs.get("token_budget") or "1")
    max_steps = str(kwargs.get("max_steps") or "900")
    gen_profile = str(kwargs.get("gen_profile") or "")
    synth_iter = str(kwargs.get("synth_iter") or "")
    rs_search_steps = str(kwargs.get("rs_search_steps") or "200")
    cars_search_steps = str(kwargs.get("cars_search_steps") or "200")

    candidates: list[Path] = []
    resolved = resolve_baseline_json_path(
        baselines_dir,
        eval_model=model,
        benchmark=benchmark,
        strategy=strategy,
        token_budget=token_budget,
        max_steps=max_steps,
        smiles_class=smiles_class,
        rs_search_steps=rs_search_steps,
        cars_search_steps=cars_search_steps,
        gen_profile=gen_profile,
        synth_iter=synth_iter,
    )
    if resolved.is_file() and _matches_name_filters(resolved, **kwargs):
        candidates.append(resolved)

    # Legacy glob fallback for partially migrated trees.
    legacy_dir = baselines_dir / strategy / _slugify(model)
    if legacy_dir.is_dir():
        candidates.extend(
            path
            for path in legacy_dir.glob(f"{benchmark_key}__*.json")
            if _matches_name_filters(path, **kwargs)
        )

    candidates = sorted(set(candidates), key=lambda p: p.stat().st_mtime, reverse=True)
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
    repo_root: Path | None = None,
    tracked_relpaths: frozenset[str] | None = None,
    **kwargs: Any,
) -> dict[str, Any] | None:
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

    for run_dir in iter_generated_run_dirs(generated_dir, name_prefix=prefix):
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
    baselines_dir: Path | None = None,
    repo_root: Path | None = None,
    tracked_relpaths: frozenset[str] | None = None,
    **kwargs: Any,
) -> dict[str, Any] | None:
    """Find the most recent metadecode success report matching the given criteria."""
    model_slug = _slugify(model)
    prefix = f"metadecode_{benchmark}_{model_slug}"
    best: Path | None = None
    best_mtime = -1.0

    for run_dir in iter_generated_run_dirs(
        generated_dir,
        eval_model=model,
        benchmark=benchmark,
        strategy="metadecode",
        name_prefix=prefix,
    ):
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
    repo_root: Path | None = None,
    tracked_relpaths: frozenset[str] | None = None,
) -> dict[str, Any] | None:
    class_suffix = ""
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
        "rs": "Reject.",
        "metadecode": r"\Tool",
    }.get(strategy, strategy)


def _main_table_rows_raw(
    baselines_dir: Path,
    generated_dir: Path,
    *,
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
                        gen_profile=DEFAULT_MAIN_GEN_PROFILE,
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
    paper_multirow: bool = False,
    bold_best: bool = False,
    repo_root: Path | None = None,
    tracked_relpaths: frozenset[str] | None = None,
) -> str:
    lines: list[str] = []
    raw_blocks = _main_table_rows_raw(
        baselines_dir,
        generated_dir,
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
    repo_root: Path | None = None,
    tracked_relpaths: frozenset[str] | None = None,
) -> str:
    lines: list[str] = []
    ablation_model = "Qwen/Qwen2.5-Coder-7B-Instruct"

    for strategy in ["gcd", "crane", "itergen", "rs", "metadecode"]:
        strategy_label = {
            "gcd": r"\GCD",
            "crane": r"\Crane",
            "itergen": r"\IterGen",
            "rs": "Reject.",
            "metadecode": r"\Tool",
        }.get(strategy, strategy)

        gsm_cells: list[str] = []
        spider_cells: list[str] = []
        for ms in step_budgets:
            for benchmark, cells in [("gsm_symbolic", gsm_cells), ("spider", spider_cells)]:
                if strategy == "metadecode":
                    data = _load_metadecode(
                        generated_dir,
                        benchmark,
                        ablation_model,
                        gen_profile=DEFAULT_MAIN_GEN_PROFILE,
                        max_steps=ms,
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
                        repo_root=repo_root,
                        tracked_relpaths=tracked_relpaths,
                    )
                cells.append(_fmt(data.get("accuracy") if data else None))

        all_cells = " & ".join(gsm_cells + spider_cells)
        lines.append(f"{strategy_label:<12s} & {all_cells} \\\\")

    return "\n".join(lines)


def emit_synth_iter_table(
    generated_dir: Path,
    synth_iters: list[int],
    *,
    baselines_dir: Path | None = None,
    repo_root: Path | None = None,
    tracked_relpaths: frozenset[str] | None = None,
) -> str:
    lines_verif: list[str] = []
    lines_acc: list[str] = []
    ablation_model = "Qwen/Qwen2.5-Coder-7B-Instruct"

    for benchmark in ["gsm_symbolic", "spider"]:
        verif_cells: list[str] = []
        acc_cells: list[str] = []
        for k in synth_iters:
            data = _load_metadecode(
                generated_dir,
                benchmark,
                ablation_model,
                gen_profile=DEFAULT_MAIN_GEN_PROFILE,
                synth_iter=k,
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
    repo_root: Path | None = None,
    tracked_relpaths: frozenset[str] | None = None,
) -> str:
    lines: list[str] = []
    ablation_model = "Qwen/Qwen2.5-Coder-7B-Instruct"
    labels = {
        "gpt5.5": r"\SynthGPT",
        "sonnet4.6": r"\SynthSonnet",
        "gemini": r"\SynthGemini",
    }

    for profile in gen_profiles:
        cells: list[str] = []
        for benchmark in ["gsm_symbolic", "spider"]:
            data = _load_metadecode(
                generated_dir,
                benchmark,
                ablation_model,
                gen_profile=profile,
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


MAIN_RESULTS_BEGIN = "% BEGIN:main_results_table_body"
MAIN_RESULTS_END = "% END:main_results_table_body"


def write_main_results_to_paper(
    body: str,
    *,
    paper_experiments: Path | None = None,
) -> Path:
    """Replace the auto-generated Table~1 body in paper/experiments.tex."""
    repo = _repo_root()
    tex_path = paper_experiments or (repo / "paper" / "experiments.tex")
    text = tex_path.read_text(encoding="utf-8")
    if MAIN_RESULTS_BEGIN not in text or MAIN_RESULTS_END not in text:
        raise SystemExit(
            f"Missing {MAIN_RESULTS_BEGIN!r} / {MAIN_RESULTS_END!r} markers in {tex_path}"
        )
    indented = "\n".join(body.splitlines())
    replacement = (
        f"{MAIN_RESULTS_BEGIN}\n"
        f"{indented}\n"
        f"{MAIN_RESULTS_END}"
    )
    start = text.find(MAIN_RESULTS_BEGIN)
    end = text.find(MAIN_RESULTS_END, start)
    if start < 0 or end < 0:
        raise SystemExit(f"Could not locate main results block in {tex_path}")
    end += len(MAIN_RESULTS_END)
    new_text = text[:start] + replacement + text[end:]
    tex_path.write_text(new_text, encoding="utf-8")
    return tex_path


def main() -> None:
    repo = _repo_root()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baselines-dir", type=Path, default=repo / "outputs" / "baselines")
    p.add_argument("--generated-dir", type=Path, default=repo / "outputs" / "generated")
    p.add_argument("--step-budgets", default="256,512,1024")
    p.add_argument("--synth-iters", default="3,5,10")
    p.add_argument(
        "--gen-profiles",
        default=",".join(DEFAULT_ABLATION_GEN_PROFILES),
        help="Synthesizer profiles for Ablation C table (default: sonnet4.6,gpt5.5).",
    )
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
    p.add_argument(
        "--write-paper",
        action="store_true",
        help="Write Table 1 body into paper/experiments.tex (implies --paper-main-table --paper-bold-best)",
    )
    p.add_argument(
        "--paper-experiments",
        type=Path,
        default=repo / "paper" / "experiments.tex",
        help="Target experiments.tex for --write-paper",
    )
    args = p.parse_args()
    if args.write_paper:
        args.paper_main_table = True
        args.paper_bold_best = True

    step_budgets = [int(x) for x in args.step_budgets.split(",")]
    synth_iters = [int(x) for x in args.synth_iters.split(",")]
    gen_profiles = [x.strip() for x in args.gen_profiles.split(",")]
    beam_sizes = [int(x) for x in args.beam_sizes.split(",")]
    helper_policies = [x.strip() for x in args.helper_policies.split(",") if x.strip()]

    tracked = _git_tracked_json_relpaths(repo) if args.git_tracked_only else None
    repo_for_track = repo if tracked is not None else None

    print("=" * 60)
    print("Table 1: Main Results")
    print("=" * 60)
    main_table_body = emit_main_table(
        args.baselines_dir,
        args.generated_dir,
        paper_multirow=args.paper_main_table,
        bold_best=args.paper_bold_best,
        repo_root=repo_for_track,
        tracked_relpaths=tracked,
    )
    print(main_table_body)
    if args.write_paper:
        if not args.paper_main_table:
            raise SystemExit("--write-paper requires --paper-main-table")
        out_path = write_main_results_to_paper(
            main_table_body,
            paper_experiments=args.paper_experiments,
        )
        print(f"\nWrote Table 1 body to {out_path}")
    print()

    print("=" * 60)
    print("Table 2: Step Budget Ablation")
    print("=" * 60)
    print(
        emit_step_budget_table(
            args.baselines_dir,
            args.generated_dir,
            step_budgets,
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
            repo_root=repo_for_track,
            tracked_relpaths=tracked,
        )
    )


if __name__ == "__main__":
    main()
