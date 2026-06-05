"""Rejection-sampling baseline: temperature-1 SynCode until syntax-valid."""

from __future__ import annotations

import argparse

ADAPTER_ID = "rs_syncode"


def run(args: argparse.Namespace) -> int:
    from synthesis.evaluate.run_legacy_fixed_strategy import run_rs_legacy_adapter

    return run_rs_legacy_adapter(args)
