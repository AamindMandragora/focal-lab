"""
ConstrainedDecoding-py: Python runtime for verified constrained decoding strategies.

This package provides Python implementations of the extern functions defined
in VerifiedAgentSynthesis.dfy, enabling execution of Dafny-compiled strategies.
"""

from .runtime_stubs import (
    LM,
    Parser,
    AllowedNext,
    ChooseToken,
    ApplyRepair,
    CheckSemantic,
    CompletePrefixConstrained,
    ApplySeqOp,
    CheckOutput,
    RunAttempt,
    RunStrategy,
    Run,
    ConstrainedDecode,
)

__all__ = [
    "LM",
    "Parser",
    "AllowedNext",
    "ChooseToken",
    "ApplyRepair",
    "CheckSemantic",
    "CompletePrefixConstrained",
    "ApplySeqOp",
    "CheckOutput",
    "RunAttempt",
    "RunStrategy",
    "Run",
    "ConstrainedDecode",
]

