#!/usr/bin/env python3
"""Plot accuracy and runtime vs max-steps from cached baseline JSON artifacts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from synthesis.project_paths import resolve_baseline_json_path, slugify


def _load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _metric(record: dict | None, key: str, default: float = 0.0) -> float:
    if not record:
        return default
    if key in record and record[key] is not None:
        return float(record[key])
    metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
    if key in metrics and metrics[key] is not None:
        return float(metrics[key])
    return default


def collect_step_budget_series(
    baseline_root: Path,
    *,
    eval_model: str,
    benchmark: str,
    strategy: str,
    token_budget: str,
    step_budgets: list[int],
) -> dict[str, list[float | None]]:
    accuracy: list[float | None] = []
    syntax: list[float | None] = []
    mean_runtime: list[float | None] = []
    for ms in step_budgets:
        path = resolve_baseline_json_path(
            baseline_root,
            eval_model=eval_model,
            benchmark=benchmark,
            strategy=strategy,
            token_budget=token_budget,
            max_steps=str(ms),
        )
        record = _load_json(path)
        accuracy.append(_metric(record, "accuracy") if record else None)
        syntax.append(_metric(record, "syntax_rate") if record else None)
        mean_runtime.append(
            _metric(record, "mean_generation_seconds_per_example") if record else None
        )
    return {
        "accuracy": accuracy,
        "syntax_rate": syntax,
        "mean_generation_seconds_per_example": mean_runtime,
    }


def plot_series(
    *,
    baseline_root: Path,
    eval_models: list[str],
    benchmarks: list[str],
    strategies: list[str],
    token_budget: str,
    step_budgets: list[int],
    output_dir: Path,
) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plotting; install it in the eval conda env"
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for benchmark in benchmarks:
        for strategy in strategies:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
            fig.suptitle(f"{benchmark} / {strategy} (tb{token_budget})")

            for eval_model in eval_models:
                series = collect_step_budget_series(
                    baseline_root,
                    eval_model=eval_model,
                    benchmark=benchmark,
                    strategy=strategy,
                    token_budget=token_budget,
                    step_budgets=step_budgets,
                )
                label = slugify(eval_model).replace("_", " ")
                axes[0].plot(
                    step_budgets,
                    series["accuracy"],
                    marker="o",
                    label=label,
                )
                axes[1].plot(
                    step_budgets,
                    series["mean_generation_seconds_per_example"],
                    marker="o",
                    label=label,
                )

            axes[0].set_xlabel("max_steps")
            axes[0].set_ylabel("accuracy")
            axes[0].set_ylim(0.0, 1.0)
            axes[0].grid(True, alpha=0.3)
            axes[0].legend(fontsize=8)

            axes[1].set_xlabel("max_steps")
            axes[1].set_ylabel("mean_generation_seconds_per_example")
            axes[1].grid(True, alpha=0.3)
            axes[1].legend(fontsize=8)

            out = output_dir / f"{slugify(benchmark)}__{strategy}__tb{token_budget}.png"
            fig.tight_layout()
            fig.savefig(out, dpi=150)
            plt.close(fig)
            written.append(out)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=Path("outputs/baselines"),
    )
    parser.add_argument(
        "--models",
        default=(
            "Qwen/Qwen3.5-2B,"
            "Qwen/Qwen3.5-4B,"
            "Qwen/Qwen3.5-9B,"
            "meta-llama/Llama-3.1-8B-Instruct"
        ),
    )
    parser.add_argument("--benchmarks", default="gsm_symbolic,spider")
    parser.add_argument(
        "--strategies",
        default="unconstrained,gcd,crane,itergen,rs,cars",
    )
    parser.add_argument("--token-budget", default="1")
    parser.add_argument("--step-budgets", default="256,512,900,1024")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/plots/step_budgets"))
    args = parser.parse_args()

    step_budgets = [int(x.strip()) for x in args.step_budgets.split(",") if x.strip()]
    eval_models = [m.strip() for m in args.models.split(",") if m.strip()]
    benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]

    written = plot_series(
        baseline_root=args.baseline_root,
        eval_models=eval_models,
        benchmarks=benchmarks,
        strategies=strategies,
        token_budget=args.token_budget,
        step_budgets=step_budgets,
        output_dir=args.output_dir,
    )
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
