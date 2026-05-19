"""Lark grammars for ``<<``-prefixed constrained decoding with a closing ``>>``."""

from __future__ import annotations

import re

# syncode entry + primary start rules that accept the answer body before ``>>``.
_START_RULE_CLOSERS: tuple[tuple[str, str], ...] = (
    (r"^start:\s*any_expr\s*$", 'start: any_expr ">>"'),
    (r"^start:\s*sql_stmt\s*$", 'start: sql_stmt ">>"'),
    (r"^start:\s*smiles\s*$", 'start: smiles ">>"'),
)

_CSD_START_CLOSERS: tuple[tuple[str, str], ...] = (
    (r"^csd_start:\s*sql_stmt\s+EOQ\s*$", 'csd_start: sql_stmt ">>"'),
    (r"^csd_start:\s*sql_stmt\s*$", 'csd_start: sql_stmt ">>"'),
)


def build_delimited_span_grammar(base_grammar: str) -> str:
    """Grammar variant when the prompt already ends with ``<<``.

    The opening delimiter is in the prompt; decoding must emit the answer then ``>>``.
    """
    text, _ = re.subn(
        r'^syncode:\s*"<<" start ">>"\s*$',
        'syncode: start ">>"',
        base_grammar,
        count=1,
        flags=re.MULTILINE,
    )
    if re.search(r"^syncode:\s*start\s*$", text, flags=re.MULTILINE):
        text = re.sub(
            r"^syncode:\s*start\s*$",
            'syncode: start ">>"',
            text,
            count=1,
            flags=re.MULTILINE,
        )

    for pattern, replacement in _START_RULE_CLOSERS:
        text, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
        if count:
            break

    for pattern, replacement in _CSD_START_CLOSERS:
        text, _ = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)

    return text


# Back-compat alias used across the repo.
build_gcd_span_grammar = build_delimited_span_grammar
