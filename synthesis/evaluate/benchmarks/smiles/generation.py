"""Generation wrapper for SMILES tasks.

Strategies decide their own constraint mode: CRANE uses << >> delimiters
naturally, GCD forces constrained from the first token via its Dafny code.
"""

from __future__ import annotations

from synthesis.evaluate.benchmarks.gsm_symbolic.generation import (
    dafny_seq_to_str,
    run_crane_csd,
    run_unconstrained,
)


__all__ = ["dafny_seq_to_str", "run_crane_csd", "run_unconstrained"]
