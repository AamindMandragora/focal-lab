"""Unconstrained baseline: legacy CRANE ``main.py`` (original mode) or Spider vLLM path."""

from __future__ import annotations

import argparse

ADAPTER_ID = "unconstrained_legacy"


def run(args: argparse.Namespace) -> int:
    from synthesis.evaluate.run_legacy_fixed_strategy import run_crane_legacy_adapter

    return run_crane_legacy_adapter(args)
