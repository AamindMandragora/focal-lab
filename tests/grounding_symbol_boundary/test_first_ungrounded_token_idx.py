"""Tests for symbol-boundary grounding rollback token indexing."""
from synthesis.evaluate.benchmarks.common.model_utils import (
    _candidate_identifiers,
    _candidate_identifiers_with_pos,
    _first_ungrounded_token_idx,
)


def test_deep_out_of_schema_name_targets_its_token_not_zero():
    support = {"singer", "name", "age"}
    tokens = ["SELECT", " name", " FROM", " badtable"]
    assert _first_ungrounded_token_idx(tokens, support) == (True, 3)


def test_fully_grounded_query_not_found():
    support = {"singer", "name", "age"}
    tokens = ["SELECT", " name", " FROM", " singer"]
    assert _first_ungrounded_token_idx(tokens, support) == (False, 0)


def test_empty_support_is_noop():
    assert _first_ungrounded_token_idx(["SELECT", " x"], set()) == (False, 0)


def test_bad_ident_after_quoted_literal_uses_correct_token():
    support = {"foo"}
    tokens = ["x = 'aaaa' ", "bar"]
    assert _first_ungrounded_token_idx(tokens, support) == (True, 1)


def test_quoted_value_is_not_an_identifier():
    support = {"singer", "name", "city"}
    tokens = ["SELECT name FROM singer WHERE city = 'london'"]
    assert _first_ungrounded_token_idx(tokens, support) == (False, 0)


def test_first_out_of_schema_identifier_wins():
    support = {"name"}
    tokens = ["SELECT", " name", ",", " badA", ",", " badB", " FROM", " t"]
    assert _first_ungrounded_token_idx(tokens, support) == (True, 3)


def test_with_pos_names_match_candidate_identifiers():
    sample = "SELECT T1.name, T2.age FROM singer AS T1 JOIN concert AS T2 WHERE city = 'NY'"
    names_pos = [n for n, _ in _candidate_identifiers_with_pos(sample)]
    names_bool = _candidate_identifiers(sample)
    assert names_pos == names_bool
