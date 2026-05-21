"""Lark grammars for delimited vs fully constrained (undelimited) decoding."""

from __future__ import annotations

import re

# Delimited: prompt ends with ``<<``; decode body then closing ``>>`` (tier-2 / CSD paths).
_START_RULE_CLOSERS: tuple[tuple[str, str], ...] = (
    (r"^start:\s*any_expr\s*$", 'start: any_expr ">>"'),
    (r"^start:\s*sql_stmt\s*$", 'start: sql_stmt ">>"'),
    (r"^start:\s*smiles\s*$", 'start: smiles ">>"'),
)

_CSD_START_CLOSERS: tuple[tuple[str, str], ...] = (
    (r"^csd_start:\s*sql_stmt\s+EOQ\s*$", 'csd_start: sql_stmt ">>"'),
    (r"^csd_start:\s*sql_stmt\s*$", 'csd_start: sql_stmt ">>"'),
    (r"^csd_start:\s*smiles\s+\">>\"\s*$", 'csd_start: smiles ">>"'),
    (r"^csd_start:\s*smiles\s*$", 'csd_start: smiles ">>"'),
)

_CONSTRAINED_START_RULES: tuple[tuple[str, str, str], ...] = (
    (r"^start:\s*any_expr\s*\">>\"\s*$", "any_expr", "start: {body}"),
    (r"^start:\s*any_expr\s*$", "any_expr", "start: {body}"),
    (r"^start:\s*sql_stmt\s*\">>\"\s*$", "sql_stmt", "start: sql_stmt"),
    (r"^start:\s*sql_stmt\s*$", "sql_stmt", "start: sql_stmt"),
    (r"^start:\s*smiles\s*\">>\"\s*$", "smiles", "start: smiles"),
    (r"^start:\s*smiles\s*$", "smiles", "start: smiles"),
)


def build_delimited_span_grammar(base_grammar: str) -> str:
    """Grammar when the prompt ends with ``<<`` and decoding must close with ``>>``."""
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


# Back-compat alias used across the repo (CSD / tier-2 delimited paths).
build_gcd_span_grammar = build_delimited_span_grammar


def build_constrained_body_grammar(
    base_grammar: str,
    *,
    require_symbolic: bool = True,
) -> str:
    """Grammar for tier-1 legacy baselines: full output is the constrained body (no ``<<`` / ``>>``)."""
    text = base_grammar
    text, _ = re.subn(
        r'^syncode:\s*.*$',
        "syncode: start",
        text,
        count=1,
        flags=re.MULTILINE,
    )

    body = "s_expr" if require_symbolic else "n_expr"
    for pattern, _kind, replacement in _CONSTRAINED_START_RULES:
        if "{body}" in replacement:
            repl = replacement.format(body=body)
        else:
            repl = replacement
        text, count = re.subn(pattern, repl, text, count=1, flags=re.MULTILINE)
        if count:
            break

    for pattern, replacement in _CSD_START_CLOSERS:
        text, _ = re.subn(
            pattern,
            replacement.replace(' ">>"', ""),
            text,
            count=1,
            flags=re.MULTILINE,
        )

    text, _ = re.subn(
        r'^csd_start:\s*any_expr\s*">>"\s*$',
        "csd_start: any_expr",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    text, _ = re.subn(
        r'^expr_only:\s*any_expr\s*">>"\s*$',
        "expr_only: any_expr",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    return text


# Back-compat alias (CARS adapter imports).
build_cars_open_span_grammar = build_constrained_body_grammar
