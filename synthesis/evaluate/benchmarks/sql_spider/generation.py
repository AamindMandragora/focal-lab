"""Generation methods for SQL Spider evaluation."""

from __future__ import annotations

from synthesis.evaluate.benchmarks.common.dafny_tokens import dafny_seq_to_str
from synthesis.evaluate.benchmarks.gsm_symbolic.generation import run_crane_csd as _run_crane_csd


def run_crane_csd(*args, **kwargs):
    """Run Spider CSDs using the visible <<...>> constrained-span surface."""
    return _run_crane_csd(*args, **kwargs)

__all__ = ["dafny_seq_to_str", "run_crane_csd"]
