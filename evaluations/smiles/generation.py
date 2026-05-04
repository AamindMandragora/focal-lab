"""Generation wrapper for SMILES tasks.

The full completion is parser-governed from the first token, but delimiter
tokens are not part of the user-visible SMILES output.
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
