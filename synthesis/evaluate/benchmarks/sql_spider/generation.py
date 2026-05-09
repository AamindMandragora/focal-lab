"""
Generation methods for SQL Spider evaluation.

Delegates to the dataset-agnostic CSD runners in gsm_symbolic.generation
so SQL shares the same Dafny strategy plumbing.  Strategies decide their own
constraint mode: CRANE uses << >> delimiters naturally, GCD forces constrained
from the first token via its Dafny code.
"""

from __future__ import annotations

from synthesis.evaluate.benchmarks.gsm_symbolic.generation import (
    dafny_seq_to_str,
    run_crane_csd,
    run_unconstrained,
)


__all__ = ["dafny_seq_to_str", "run_crane_csd", "run_unconstrained"]
