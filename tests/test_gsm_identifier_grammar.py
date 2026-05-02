from pathlib import Path

from lark import Lark


def test_gsm_grammar_accepts_descriptive_identifiers():
    grammar = Path("grammars/gsm.lark").read_text()
    parser = Lark(grammar, start="start", parser="lalr")

    parser.parse("o_new")
    parser.parse("brokerage_fee")
    parser.parse("total_cost + brokerage_fee")
