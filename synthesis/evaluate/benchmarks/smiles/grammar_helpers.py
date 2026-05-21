"""SMILES Lark grammar transforms for tier-1 body vs tier-2 delimited CoT."""

from __future__ import annotations

from synthesis.evaluate.benchmarks.common.delimiter_grammar import (
    build_constrained_body_grammar,
    build_delimited_span_grammar,
)


def build_smiles_tier1_body_grammar(base_grammar: str) -> str:
    """Tier-1 (GCD / IterGen / CARS): entire decode is one SMILES string, no ``<<`` / ``>>``."""
    return build_constrained_body_grammar(base_grammar, require_symbolic=False)


def build_smiles_tier2_delimited_grammar(base_grammar: str) -> str:
    """
    Tier-2 (Unconstrained / CRANE): free-form reasoning, then ``<<`` SMILES ``>>``.

    The closing ``>>`` is a grammar literal on ``start`` / ``csd_start``, not a SMILES token.
    """
    return build_delimited_span_grammar(base_grammar)


__all__ = [
    "build_smiles_tier1_body_grammar",
    "build_smiles_tier2_delimited_grammar",
]
