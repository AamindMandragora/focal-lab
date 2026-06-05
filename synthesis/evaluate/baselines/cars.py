"""CARS baseline: patched ``legacy/cars`` oracle rejection sampling."""

from __future__ import annotations

import argparse

ADAPTER_ID = "cars_legacy_cars"


def run(args: argparse.Namespace) -> int:
    from synthesis.evaluate.run_legacy_fixed_strategy import run_cars_legacy_adapter

    return run_cars_legacy_adapter(args)
