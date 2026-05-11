#!/usr/bin/env python3
"""
Collect baseline and synthesis results into paper-ready LaTeX table fragments.

Reads baseline JSONs from outputs/baselines/ and metadecode success reports
from outputs/generated/, then emits:
  1. Main results table (Table 1)
  2. Step-budget ablation (Table 2)
  3. Synthesis-iterations ablation (Table 3)
  4. Synthesizer-model ablation (Table 4)

Usage:
  python -m synthesis.scripts.collect_paper_results
  python -m synthesis.scripts.collect_paper_results --baselines-dir outputs/baselines --generated-dir outputs/generated
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


MODELS = [
    ("Qwen/Qwen2.5-Coder-1.5B-Instruct", "1.5B"),
    ("Qwen/Qwen2.5-Coder-7B-Instruct", "7B"),
    ("Qwen/Qwen2.5-Coder-14B-Instruct", "14B"),
    ("meta-llama/Llama-3.1-8B-Instruct", "Llama-8B"),
]
BENCHMARKS = ["gsm_symbolic", "spider", "smiles"]
STRATEGIES = ["unconstrained", "gcd", "crane", "itergen", "cars", "metadecode"]


def _slugify(s: str) -> str:
    return s.replace("/", "_").replace(":", "_").replace(" ", "_").replace("-", "_")


def _load_baseline(baselines_dir: Path, strategy: str, model: str, benchmark: str) -> dict[str, Any] | None:
    model_slug = _slugify(model)
    candidates = sorted(
        (baselines_dir / strategy / model_slug).glob(f"{benchmark}__*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ) if (baselines_dir / strategy / model_slug).is_dir() else []
    if not candidates:
        return None
    try:
        return json.loads(candidates[0].read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _load_metadecode(generated_dir: Path, benchmark: str, model: str, **kwargs: Any) -> dict[str, Any] | None:
    """Find the most recent metadecode success report matching the given criteria."""
    model_slug = _slugify(model)
    prefix = f"metadecode_{benchmark}_{model_slug}"
    best: Path | None = None
    best_mtime = -1.0

    for run_dir in generated_dir.iterdir():
        if not run_dir.is_dir() or not run_dir.name.startswith(prefix):
            continue
        success = run_dir / "results" / "success_report.json"
        if not success.is_file():
            continue

        name = run_dir.name
        match = True
        for key, value in kwargs.items():
            if key == "synth_iter":
                if f"_iter{value}_" not in name and not name.endswith(f"_iter{value}"):
                    match = False
            elif key == "gen_profile":
                gen_slug = _slugify(str(value))
                if gen_slug not in name:
                    match = False
            elif key == "max_steps":
                if f"_ms{value}_" not in name and not name.endswith(f"_ms{value}"):
                    match = False
        if not match:
            continue

        mtime = success.stat().st_mtime
        if mtime > best_mtime:
            best = success
            best_mtime = mtime

    if best is None:
        return None
    try:
        report = json.loads(best.read_text())
        ev = report.get("evaluation_result") or {}
        return {
            "accuracy": ev.get("accuracy", 0.0),
            "syntax_rate": ev.get("syntax_rate", 0.0),
        }
    except (json.JSONDecodeError, OSError):
        return None


def _fmt(value: float | None, pct: bool = True) -> str:
    if value is None:
        return r"\todo{--}"
    v = value * 100 if pct else value
    return f"{v:.0f}"


def emit_main_table(baselines_dir: Path, generated_dir: Path) -> str:
    lines: list[str] = []
    for model_full, model_short in MODELS:
        for strategy in STRATEGIES:
            row_cells: list[str] = []
            for benchmark in BENCHMARKS:
                if strategy == "metadecode":
                    data = _load_metadecode(generated_dir, benchmark, model_full)
                else:
                    data = _load_baseline(baselines_dir, strategy, model_full, benchmark)
                if data is None:
                    row_cells.extend([r"\todo{--}", r"\todo{--}"])
                else:
                    row_cells.append(_fmt(data.get("accuracy")))
                    row_cells.append(_fmt(data.get("syntax_rate")))

            strategy_label = {
                "unconstrained": "Unconstr.",
                "gcd": r"\GCD",
                "crane": r"\Crane",
                "itergen": r"\IterGen",
                "cars": r"\CARS",
                "metadecode": r"\Tool",
            }.get(strategy, strategy)
            cells = " & ".join(row_cells)
            lines.append(f"& {strategy_label:<12s} & {cells} \\\\")
        lines.append(r"\midrule")

    if lines and lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"

    return "\n".join(lines)


def emit_step_budget_table(baselines_dir: Path, generated_dir: Path, step_budgets: list[int]) -> str:
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
                    data = _load_metadecode(generated_dir, benchmark, ablation_model, max_steps=ms)
                else:
                    data = _load_baseline(baselines_dir, strategy, ablation_model, benchmark)
                cells.append(_fmt(data.get("accuracy") if data else None))

        all_cells = " & ".join(gsm_cells + spider_cells + smiles_cells)
        lines.append(f"{strategy_label:<12s} & {all_cells} \\\\")

    return "\n".join(lines)


def emit_synth_iter_table(generated_dir: Path, synth_iters: list[int]) -> str:
    lines_verif: list[str] = []
    lines_acc: list[str] = []
    ablation_model = "Qwen/Qwen2.5-Coder-7B-Instruct"

    for benchmark in ["gsm_symbolic", "spider", "smiles"]:
        verif_cells: list[str] = []
        acc_cells: list[str] = []
        for k in synth_iters:
            data = _load_metadecode(generated_dir, benchmark, ablation_model, synth_iter=k)
            acc_cells.append(_fmt(data.get("accuracy") if data else None))
            verif_cells.append(r"\todo{--}")
        lines_verif.extend(verif_cells)
        lines_acc.extend(acc_cells)

    verif_row = r"Verif.\ pass rate (\%) & " + " & ".join(lines_verif) + r" \\"
    acc_row = r"Accuracy (\%)          & " + " & ".join(lines_acc) + r" \\"
    return verif_row + "\n" + acc_row


def emit_synth_model_table(generated_dir: Path, gen_profiles: list[str]) -> str:
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
                generated_dir, benchmark, ablation_model, gen_profile=profile
            )
            cells.append(r"\todo{--}")  # verif pass rate
            cells.append(_fmt(data.get("accuracy") if data else None))
        label = labels.get(profile, profile)
        all_cells = " & ".join(cells)
        lines.append(f"{label:<16s} & {all_cells} \\\\")

    return "\n".join(lines)


def main() -> None:
    repo = _repo_root()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baselines-dir", type=Path, default=repo / "outputs" / "baselines")
    p.add_argument("--generated-dir", type=Path, default=repo / "outputs" / "generated")
    p.add_argument("--step-budgets", default="256,512,1024")
    p.add_argument("--synth-iters", default="3,5,10")
    p.add_argument("--gen-profiles", default="gpt5.4,opus4.7,gemini-pro")
    args = p.parse_args()

    step_budgets = [int(x) for x in args.step_budgets.split(",")]
    synth_iters = [int(x) for x in args.synth_iters.split(",")]
    gen_profiles = [x.strip() for x in args.gen_profiles.split(",")]

    print("=" * 60)
    print("Table 1: Main Results")
    print("=" * 60)
    print(emit_main_table(args.baselines_dir, args.generated_dir))
    print()

    print("=" * 60)
    print("Table 2: Step Budget Ablation")
    print("=" * 60)
    print(emit_step_budget_table(args.baselines_dir, args.generated_dir, step_budgets))
    print()

    print("=" * 60)
    print("Table 3: Synthesis Iterations Ablation")
    print("=" * 60)
    print(emit_synth_iter_table(args.generated_dir, synth_iters))
    print()

    print("=" * 60)
    print("Table 4: Synthesizer Model Ablation")
    print("=" * 60)
    print(emit_synth_model_table(args.generated_dir, gen_profiles))


if __name__ == "__main__":
    main()
