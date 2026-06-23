"""Confirm SymbolPosMap records table_ref/column_ref with positions."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GRAMMARS = REPO / "synthesis" / "evaluate" / "grammars"
SYNCODE_DIR = REPO / "synthesis" / "evaluate" / "syncode"
if str(SYNCODE_DIR) in sys.path:
    sys.path.remove(str(SYNCODE_DIR))
sys.path.insert(0, str(SYNCODE_DIR))

from syncode.parsers.grammars import Grammar
from syncode.parsers import create_base_parser
from syncode.parsers.incremental_parser import IncrementalParser


def _count(spm, name):
    return spm.get_symbol_count(name)


def test_symbol_pos_map_records_schema_symbols():
    text = (GRAMMARS / "sql.lark").read_text()
    grammar = Grammar(text)
    base = create_base_parser(grammar)

    # 1) A complete query: confirm table_ref + column_ref are recorded with spans.
    ip = IncrementalParser(base, ignore_whitespace=False)
    q = "SELECT name FROM singer WHERE age > 20"
    ip.get_acceptable_next_terminals(q)
    spm = ip.symbol_pos_map
    recorded = {k: list(v) for k, v in spm._pos_map.items() if v}
    print("[recorded symbols on full query]")
    for k in sorted(recorded):
        print(f"  {k}: {recorded[k]}")

    table_ct = _count(spm, "table_ref")
    column_ct = _count(spm, "column_ref")
    print(f"\ntable_ref count = {table_ct}")
    print(f"column_ref count = {column_ct}")

    assert table_ct >= 1, "FAIL: no table_ref recorded — symbol name is wrong"
    assert column_ct >= 1, "FAIL: no column_ref recorded — symbol name is wrong"

    # 2) Monotonic rise mid-query: drive growing prefixes through a FRESH parser
    #    (cumulative, like the runtime) and confirm the combined count is
    #    non-decreasing and ends > 0. This is exactly the signal the boundary uses.
    ip2 = IncrementalParser(base, ignore_whitespace=False)
    prefixes = [
        "SELECT",
        "SELECT name",
        "SELECT name FROM",
        "SELECT name FROM singer",
        "SELECT name FROM singer WHERE",
        "SELECT name FROM singer WHERE age",
        "SELECT name FROM singer WHERE age > 20",
    ]
    counts = []
    for p in prefixes:
        ip2.get_acceptable_next_terminals(p)
        s = ip2.symbol_pos_map
        c = _count(s, "table_ref") + _count(s, "column_ref")
        counts.append(c)
        print(f"  prefix={p!r:45} schema_symbol_count={c}")

    assert counts == sorted(counts), f"FAIL: count not monotonic: {counts}"
    assert counts[-1] >= 2, f"FAIL: expected >=2 schema symbols at end, got {counts[-1]}"

    # 3) JOIN with aliases: more table/column refs.
    ip3 = IncrementalParser(base, ignore_whitespace=False)
    jq = "SELECT T1.name FROM singer AS T1 JOIN concert AS T2 ON T1.id = T2.singer_id"
    ip3.get_acceptable_next_terminals(jq)
    s3 = ip3.symbol_pos_map
    jt = _count(s3, "table_ref")
    jc = _count(s3, "column_ref")
    print(f"\n[join query] table_ref={jt} column_ref={jc}")
    assert jt >= 2, f"FAIL: expected >=2 table_ref on JOIN, got {jt}"

