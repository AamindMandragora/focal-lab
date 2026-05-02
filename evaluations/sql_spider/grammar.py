"""
Dynamic grammar construction for SQL Spider evaluation.

Narrows the generic TABLE_NAME / COLUMN_NAME / IDENTIFIER terminals in
grammars/sql.lark down to the concrete identifiers in the current
example's schema, mirroring the role of
evaluations.gsm_symbolic.grammar.build_dynamic_grammar.
"""

from __future__ import annotations

import re
from typing import Dict, List, Set, Tuple


def parse_db_info(db_info: str) -> Dict[str, List[str]]:
    """
    Parse the itergen-style db_info block into {table: [cols]}.

    The block looks like:
        # stadium ( stadium_id , location , name , capacity )
        # singer ( singer_id , name , country )
        # concert.stadium_id = stadium.stadium_id
    """
    schema: Dict[str, List[str]] = {}
    if not db_info:
        return schema

    for raw in db_info.split("#"):
        line = raw.strip()
        if not line or "(" not in line or ")" not in line:
            continue
        # Skip foreign-key lines like "concert.stadium_id = stadium.stadium_id"
        if "=" in line and "(" not in line.split("=")[0]:
            continue
        try:
            head, rest = line.split("(", 1)
            cols_str = rest.split(")", 1)[0]
        except ValueError:
            continue
        table = head.strip().lower()
        cols = [c.strip().lower() for c in cols_str.split(",") if c.strip()]
        if table and cols:
            schema[table] = cols
    return schema


def extract_schema_identifiers(db_info: str) -> Tuple[List[str], List[str]]:
    """
    Return (tables, columns) as sorted unique identifier lists from a db_info block.

    Columns contain just the bare column names (no table prefix).
    """
    schema = parse_db_info(db_info)
    tables: Set[str] = set()
    columns: Set[str] = set()
    for tbl, cols in schema.items():
        tables.add(tbl)
        for col in cols:
            columns.add(col)
    return sorted(tables), sorted(columns)


def _rule_body_literals(names: List[str]) -> str:
    # Sort descending by length so longer identifiers match before prefixes.
    ordered = sorted({n for n in names if n}, key=lambda s: (-len(s), s))
    if not ordered:
        return ""
    return " | ".join(f'"{n}"' for n in ordered)


def build_dynamic_sql_grammar(
    base_grammar: str,
    tables: List[str],
    columns: List[str],
) -> str:
    """
    Returns the base grammar unchanged.

    The unified NAME terminal in grammars/sql.lark covers tables, columns, and
    aliases as one permissive identifier — itergen-style. Schema narrowing is
    intentionally NOT applied here because narrowing creates overlapping
    literal terminals that defeat LALR disambiguation (see the Apr 2026
    investigation in memory/project_sql_grammar_fix.md). The (tables, columns)
    arguments are accepted for API compatibility with callers that still pass
    them, but they're ignored.
    """
    return base_grammar
