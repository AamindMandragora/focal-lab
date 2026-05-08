"""
Compatibility wrapper for parser utilities.

Historically the project had two near-identical parser utility modules:
`synthesis.evaluate.parser_utils` and
`synthesis.evaluate.benchmarks.common.parser_utils`.

To avoid duplicated logic and drift, this module now re-exports the canonical
implementations from `benchmarks.common.parser_utils`.
"""

from synthesis.evaluate.benchmarks.common.parser_utils import (  # noqa: F401
    create_lark_dafny_parser,
    prewarm_dfa_mask_store,
    print_parser_timings,
)

__all__ = [
    "create_lark_dafny_parser",
    "prewarm_dfa_mask_store",
    "print_parser_timings",
]
