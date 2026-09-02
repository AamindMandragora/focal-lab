"""Targeted TDD test for the span-closing (IsCompletePrefix) fairness fix.

THE BUG: GSM/SQL CSD parsers are built with start="csd_start" (e.g. gsm.lark
`csd_start: any_expr ">>"`), which REQUIRES the closing ">>".  But ">>" is
permanently grammar-masked during constrained decoding, so `_is_complete`
(which parses with that same parser) can never return True for span content ->
IsCompletePrefix is always False -> CloseSpanIfComplete never fires -> spans
stay open -> 0/0 wedge (observed on GSM-4B att7-12, and unclosed-span failures
on GSM-2B).

THE FIX: `_is_complete` must check completeness against the CONTENT rule
("start", i.e. any_expr / sql_stmt / smiles) WITHOUT the closer.  The strategy
force-appends ">>" itself via CloseConstrainedSpan.  So `_get_parser_components`
gains a 2nd Lark parser with start="start", stored as `self._complete_lark`,
and `_is_complete` uses it.

FAIRNESS INVARIANT (must stay true): the SYNTAX-RATE metric is computed by a
SEPARATE parser (start="syncode", in eval_logic) that still checks the literal
"<<...>>".  Changing `_is_complete` must NOT make ">>"-less output count as
valid syntax.

RED (current code): `_get_parser_components` returns a 3-tuple -> the
len(comps)==4 assertion fails.
GREEN (after fix):   4-tuple; the completeness parser accepts bare "42".
"""
import os
import sys

REPO = "/home/aadivyar/csd-generation"
sys.path.insert(0, REPO)

import importlib.util  # noqa: E402

# Default: import the DEPLOYED module (used for the RED run).  Set
# PARSER_UTILS_PATH to load a candidate file standalone (used for GREEN against a
# temp copy, so the deployed path stays untouched while a re-eval is running).
_PU_PATH = os.environ.get("PARSER_UTILS_PATH")
if _PU_PATH:
    _spec = importlib.util.spec_from_file_location("parser_utils_under_test", _PU_PATH)
    _pu = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_pu)
    _get_parser_components = _pu._get_parser_components
    _load_grammar_text = _pu._load_grammar_text
else:
    from synthesis.evaluate.benchmarks.common.parser_utils import (  # noqa: E402
        _get_parser_components,
        _load_grammar_text,
    )
from lark import Lark  # noqa: E402

GSM_GRAMMAR = os.path.join(REPO, "synthesis/evaluate/grammars/gsm.lark")


def _parses(parser, text):
    try:
        parser.parse(text)
        return True
    except Exception:
        return False


def main():
    grammar_text = _load_grammar_text(GSM_GRAMMAR)

    # Build components exactly as the GSM CSD caller does: start="csd_start".
    comps = _get_parser_components(grammar_text, "csd_start")

    # --- THE CHANGE: a 4th component, the completeness parser ---
    assert len(comps) == 4, (
        f"expected 4-tuple (grammar, base_parser, lark_parser, complete_lark) "
        f"after fix, got {len(comps)}-tuple"
    )
    _grammar, _base_parser, lark_parser, complete_lark = comps

    # 1) The prefix/validity parser keeps start="csd_start" -> REQUIRES ">>".
    #    Documents the deadlock: bare "42" is NOT a complete csd_start parse.
    assert _parses(lark_parser, "42") is False, \
        "csd_start parser must reject bare '42' (it requires the closing '>>')"
    assert _parses(lark_parser, "42 >>") is True, \
        "csd_start parser must accept '42 >>'"

    # 2) The completeness parser uses start (any_expr) -> bare "42" IS complete.
    #    This is the fix that lets IsCompletePrefix fire so spans can close.
    assert _parses(complete_lark, "42") is True, \
        "completeness parser must accept bare '42' (any_expr is complete without '>>')"
    assert _parses(complete_lark, "n + m") is True, \
        "completeness parser must accept a complete expression 'n + m'"
    assert _parses(complete_lark, "(n * m)") is True, \
        "completeness parser must accept '(n * m)'"
    # An incomplete expression must STILL be rejected (no premature/garbage close).
    assert _parses(complete_lark, "n +") is False, \
        "completeness parser must reject the dangling 'n +'"

    # 3) FAIRNESS: the syntax-rate metric parser (start='syncode') is unchanged
    #    and STILL requires the literal '<<' ... '>>'.  ">>"-less content is NOT
    #    counted as valid syntax, so the fix cannot inflate the syntax rate.
    metric = Lark(grammar_text, start="syncode", parser="lalr")
    assert _parses(metric, "42") is False, \
        "metric must reject bare '42' (no << >>)"
    assert _parses(metric, "<<42>>") is True, \
        "metric accepts a properly delimited '<<42>>'"
    assert _parses(metric, "42>>") is False, \
        "metric still requires the opening '<<'"

    print("ALL_ASSERTIONS_PASSED")


if __name__ == "__main__":
    main()
