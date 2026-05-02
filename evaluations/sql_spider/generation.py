"""
Generation methods for SQL Spider evaluation.

Delegates to the dataset-agnostic CSD runners in evaluations.gsm_symbolic.generation
so SQL shares the same Dafny strategy plumbing. Spider starts inside a hidden
constrained chunk: the whole SQL query is parser-governed, but no visible
<< / >> delimiters are required in the final answer.
"""

from __future__ import annotations

from evaluations.gsm_symbolic.generation import (
    dafny_seq_to_str,
    run_crane_csd as _run_crane_csd,
    run_unconstrained,
)


def run_crane_csd(*args, **kwargs):
    kwargs.setdefault("start_inside_constrained", True)
    return _run_crane_csd(*args, **kwargs)


__all__ = ["dafny_seq_to_str", "run_crane_csd", "run_unconstrained"]
