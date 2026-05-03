"""
SMILES evaluation module.

This benchmark uses single-line chemistry answers, primarily SMILES strings,
with the same CSD runtime plumbing as GSM-Symbolic.
"""

from evaluations.smiles.dataset import (
    DEFAULT_SMILES_CONFIGS,
    SmilesExample,
    load_smiles,
)
from evaluations.smiles.generation import (
    dafny_seq_to_str,
    run_crane_csd,
    run_unconstrained,
)
from evaluations.smiles.environment import (
    load_compiled_modules,
    setup_dafny_environment,
    verify_critical_tokens,
)
from evaluations.smiles.metrics import SmilesMetrics

__all__ = [
    "DEFAULT_SMILES_CONFIGS",
    "SmilesExample",
    "load_smiles",
    "dafny_seq_to_str",
    "run_crane_csd",
    "run_unconstrained",
    "load_compiled_modules",
    "setup_dafny_environment",
    "verify_critical_tokens",
    "SmilesMetrics",
]
