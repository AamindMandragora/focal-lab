"""Behavioral tests for the prompt-grounding extern (SpanGrounded) and helpers.

Run from the repo's eval root so `benchmarks.common.model_utils` imports:
    cd synthesis/evaluate
    PYTHONPATH=. python benchmarks/common/test_grounding.py

These tests pin the contract of the grounding helper used by the Dafny library
method RegenerateUnitOnGroundingFailure:
  * _parse_schema_support isolates the REAL example's schema (last db_info block)
    from the few-shot example's schema in the same prompt.
  * _candidate_identifiers strips string literals and short aliases, drops keywords.
  * SpanGrounded returns True iff every identifier-like token in the span is in
    the schema support set, and True (no-op) when the prompt has no schema.
"""

from benchmarks.common.model_utils import (
    _parse_schema_support,
    _candidate_identifiers,
    _TensorizedLMBase,
)


# A prompt that mimics the Spider chat template: the FEW-SHOT user turn carries
# the `singer` schema, the REAL user turn carries the `employee`/`department`
# schema. Grounding must use only the real (last) schema.
_PROMPT = (
    "You are given a database schema and a question.\n"
    "db_id: concert_singer\n"
    "db_info: # singer ( singer_id , name , country , age )\n"
    "question: How many singers do we have?\n"
    "SQL: << SELECT count(*) FROM singer >>\n"
    "db_id: company_1\n"
    "db_info: # employee ( emp_id , emp_name , dept )\n"
    "# department ( dept_id , dept_name )\n"
    "question: List employee names in the Sales department.\n"
)

_PROMPT_NO_SCHEMA = (
    "Solve the math word problem step by step, wrapping the final answer in << >>.\n"
    "question: What is 2 plus 2?\n"
)


class _StubLM(_TensorizedLMBase):
    """Minimal LM exposing only what SpanGrounded needs (no torch init)."""

    def __init__(self, instruction_text):
        self.instruction_text = instruction_text

    def _to_str(self, s):
        return s if isinstance(s, str) else "".join(s)


def test_parse_schema_support_isolates_real_schema():
    support = _parse_schema_support(_PROMPT)
    # Real schema identifiers present.
    for name in ("employee", "emp_id", "emp_name", "dept", "department", "dept_id", "dept_name"):
        assert name in support, f"expected {name!r} in support, got {sorted(support)}"
    # Few-shot (singer) schema identifiers must NOT leak in.
    for leaked in ("singer", "singer_id", "country", "age", "name"):
        assert leaked not in support, f"{leaked!r} leaked from few-shot schema"


def test_parse_schema_support_empty_when_no_schema():
    assert _parse_schema_support(_PROMPT_NO_SCHEMA) == set()
    assert _parse_schema_support("") == set()


def test_candidate_identifiers_strips_literals_keywords_aliases():
    cands = _candidate_identifiers("SELECT emp_name FROM employee WHERE dept = 'Sales'")
    assert "emp_name" in cands and "employee" in cands and "dept" in cands
    assert "sales" not in cands, "string-literal content should be stripped"
    assert "select" not in cands and "from" not in cands and "where" not in cands
    # Aliases (t1) and single letters dropped.
    aliased = _candidate_identifiers("SELECT t1.emp_name FROM employee AS t1")
    assert "t1" not in aliased, "alias token t1 should be dropped"
    assert "emp_name" in aliased and "employee" in aliased


def test_span_grounded_true_for_in_schema():
    lm = _StubLM(_PROMPT)
    assert lm.SpanGrounded("SELECT emp_name FROM employee") is True
    assert lm.SpanGrounded("SELECT dept_name FROM department") is True


def test_span_grounded_false_for_out_of_schema():
    lm = _StubLM(_PROMPT)
    assert lm.SpanGrounded("SELECT bogus_col FROM employee") is False
    assert lm.SpanGrounded("SELECT emp_name FROM nonexistent_table") is False


def test_span_grounded_allows_aliases_and_literals():
    lm = _StubLM(_PROMPT)
    q = "SELECT t1.emp_name FROM employee AS t1 WHERE t1.dept = 'Engineering'"
    assert lm.SpanGrounded(q) is True


def test_span_grounded_noop_without_schema():
    lm = _StubLM(_PROMPT_NO_SCHEMA)
    # No db_info -> empty support -> always grounded (no-op for non-SQL tasks).
    assert lm.SpanGrounded("anything goes here including << 6 >>") is True


def test_span_grounded_cache_is_stable():
    lm = _StubLM(_PROMPT)
    first = lm._grounding_support_set()
    second = lm._grounding_support_set()
    assert first == second and first is second


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = []
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failures.append((t.__name__, str(e)))
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    import sys
    sys.exit(_run())
