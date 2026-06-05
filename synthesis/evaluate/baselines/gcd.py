"""GCD baseline: vendored SynCode grammar-strict decoding."""

from __future__ import annotations

import argparse

ADAPTER_ID = "gcd_syncode"


def run(args: argparse.Namespace) -> int:
    from synthesis.evaluate.run_legacy_fixed_strategy import run_gcd_legacy_adapter

    return run_gcd_legacy_adapter(args)
