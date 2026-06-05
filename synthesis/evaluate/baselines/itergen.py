"""IterGen baseline: patched ``legacy/itergen`` grammar-masked decoding."""

from __future__ import annotations

import argparse

ADAPTER_ID = "itergen_legacy"


def run(args: argparse.Namespace) -> int:
    from synthesis.evaluate.run_legacy_fixed_strategy import run_itergen_legacy_adapter

    return run_itergen_legacy_adapter(args)
