"""Build a per-example schema-restricted SQL grammar.

Replaces the generic NAME terminal in sql.lark with one that only allows
table/column names from the example's db_info, plus short alias patterns.

This prevents the model from generating table or column names that don't
exist in the schema, addressing the over-joining and wrong-table-selection
errors that cap Spider-7B at ~58%.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

_BASE_GRAMMAR_PATH = Path(__file__).parent.parent.parent / "grammars" / "sql.lark"


def parse_schema_names(db_info: str) -> list[str]:
    """Extract table and column names from Spider db_info.

    db_info format (one line per table):
        # table_name ( col1 , col2 , table2.col3 , ... )

    Returns a deduplicated list in declaration order.
    """
    names: list[str] = []
    seen: set[str] = set()

    for line in db_info.splitlines():
        line = line.strip()
        if not line.startswith("#"):
            continue
        line = line[1:].strip()
        match = re.match(r"(\w+)\s*\((.+)\)", line)
        if not match:
            continue

        table_name = match.group(1)
        if table_name not in seen:
            seen.add(table_name)
            names.append(table_name)

        for col in match.group(2).split(","):
            col = col.strip()
            if "." in col:
                col = col.split(".")[-1].strip()
            if col and re.match(r"^\w+$", col) and col not in seen:
                seen.add(col)
                names.append(col)

    return names


@lru_cache(maxsize=512)
def _build_schema_grammar_cached(db_info: str) -> str:
    return _build(db_info)


def build_schema_grammar(db_info: str) -> str:
    """Return sql.lark text with NAME restricted to schema names + short aliases.

    The NAME terminal is a regex alternation (longest match first) of:
      - All schema table/column names (case-insensitive)
      - Single letter: a, b, T, s  (table/column aliases)
      - Letter + digits: T1, T2, a1  (numbered aliases)

    Using a regex (not string literals) preserves NAME as a priority-(-1) terminal,
    so inline SQL keywords like "COUNT"i (priority 0) still win when they match the
    same input — but a longer schema name (e.g. "country" 7 chars) beats a shorter
    keyword prefix ("count" 5 chars) via Lark's longest-match rule.

    The generic grammar is returned unchanged when db_info is empty.
    """
    return _build_schema_grammar_cached(db_info)


def _build(db_info: str) -> str:
    schema_names = parse_schema_names(db_info)
    base_grammar = _BASE_GRAMMAR_PATH.read_text()

    if not schema_names:
        return base_grammar

    # Sort schema names longest-first so Python regex alternation picks the
    # longest match (Python re uses first-match, so longer must come first).
    sorted_names = sorted(set(schema_names), key=len, reverse=True)

    # All schema names are \w+ so re.escape is a no-op but used defensively.
    name_alts = [re.escape(n) for n in sorted_names]

    # Short alias patterns that don't overlap with any SQL keyword:
    #   - letter+digits: T1, T2, a1, s2 (keywords never contain digits)
    #   - single letter: a, b, T, s
    name_alts += [r"[A-Za-z][0-9]+", r"[A-Za-z]"]

    combined = "(?:" + "|".join(name_alts) + ")"

    # The lookahead (?![a-zA-Z0-9_]) prevents matching a prefix of a longer
    # identifier (e.g. "sing" inside "singer"). Schema names from db_info are
    # lowercase; the model (Qwen-7B) generates lowercase SQL, so case-sensitive
    # matching is fine. Uppercase aliases like T1 are covered by [A-Za-z][0-9]+.
    name_terminal = f"NAME: /{combined}(?![a-zA-Z0-9_])/"

    new_grammar = re.sub(
        r"^NAME: /\[a-zA-Z_\]\[a-zA-Z0-9_\]\*/\s*$",
        name_terminal,
        base_grammar,
        flags=re.MULTILINE,
    )

    return new_grammar
