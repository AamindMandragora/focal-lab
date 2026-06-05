"""SMILES baselines: pooled native protocol for every fixed strategy."""

from __future__ import annotations

import argparse

ADAPTER_ID = "smiles_pooled"


def run(args: argparse.Namespace) -> int:
    from synthesis.evaluate.benchmarks.smiles.pooled_baseline import (
        run_smiles_pooled_legacy_adapter,
    )

    return run_smiles_pooled_legacy_adapter(args)
