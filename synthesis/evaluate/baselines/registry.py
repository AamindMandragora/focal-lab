"""Dispatch fixed-strategy baselines to per-strategy legacy adapters."""

from __future__ import annotations

import argparse
from typing import Callable

STRATEGIES: tuple[str, ...] = (
    "unconstrained",
    "gcd",
    "crane",
    "itergen",
    "cars",
    "rs",
)

# Adapter ids emitted in baseline JSON ``metrics.adapter`` (also used for cache validation).
ADAPTER_IDS: dict[str, str] = {
    "unconstrained": "unconstrained_legacy",
    "gcd": "gcd_syncode",
    "crane": "crane_legacy_crane",
    "itergen": "itergen_legacy",
    "cars": "cars_legacy_cars",
    "rs": "rs_syncode",
}

CRANE_ADAPTER_IDS: frozenset[str] = frozenset(
    {
        "crane_legacy_main",
        "crane_legacy_crane",
        "crane_shared_evaluator",
        "crane_repo",
    }
)


def _normalize_dataset(dataset: str) -> str:
    return "gsm_symbolic" if dataset == "gsm" else dataset


def _adapter_for(strategy: str) -> Callable[[argparse.Namespace], int]:
    key = (strategy or "").strip().lower()
    if key == "unconstrained":
        from synthesis.evaluate.baselines.unconstrained import run as run_unconstrained

        return run_unconstrained
    if key == "gcd":
        from synthesis.evaluate.baselines.gcd import run as run_gcd

        return run_gcd
    if key == "crane":
        from synthesis.evaluate.baselines.crane import run as run_crane

        return run_crane
    if key == "itergen":
        from synthesis.evaluate.baselines.itergen import run as run_itergen

        return run_itergen
    if key == "cars":
        from synthesis.evaluate.baselines.cars import run as run_cars

        return run_cars
    if key == "rs":
        from synthesis.evaluate.baselines.rs import run as run_rs

        return run_rs
    raise ValueError(
        f"Unknown baseline strategy {strategy!r}; expected one of {list(STRATEGIES)}"
    )


def run_baseline_strategy(args: argparse.Namespace) -> int:
    """Run one fixed-strategy baseline and write ``args.output_json``."""
    dataset = _normalize_dataset(args.dataset)
    strategy = str(args.strategy).strip().lower()
    if strategy not in STRATEGIES:
        raise ValueError(
            f"Unknown baseline strategy {strategy!r}; expected one of {list(STRATEGIES)}"
        )

    if dataset == "smiles":
        from synthesis.evaluate.baselines.smiles import run as run_smiles

        return run_smiles(args)

    return _adapter_for(strategy)(args)
