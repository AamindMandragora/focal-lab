"""An unreadable variable_types field must not silently change how GSM is graded.

The defect
----------
`gsm_symbolic/eval_logic.py:171-180`:

    vt = example.get("variable_types", {})
    if isinstance(vt, str):
        try:
            vt = ast.literal_eval(vt)
        except (ValueError, SyntaxError):
            vt = {}                      # <-- unreadable becomes "absent"
    if not isinstance(vt, dict):
        vt = {}
    if vt and example.get("answer_parsed"):
        return evaluator._gsm_symbolic_equivalence(actual, expected, vt)
    ...numeric comparison instead...

This is worse than swallowing an error into a wrong answer. `{}` is falsy, so
the `if vt` on the next line is False and the symbolic grader is never called.
The example is graded by a different method -- a plain numeric match -- and
nothing anywhere records that the method changed.

So a malformed dataset field does not produce a visible failure or even a
suspicious zero. It produces a number computed by a grader nobody chose, mixed
into the same average as the examples graded symbolically. The two graders do
not measure the same thing: CRANE's is a proof of algebraic equivalence over
all valid inputs, ours-by-fallback is "do these two numbers match".

This also makes the fix in `_gsm_symbolic_equivalence` unreachable on this
path. Removing the catch there stops nothing if the caller has already turned
the failure into a routing decision.

The numeric grader is gone entirely
-----------------------------------
The first version of this fix kept the numeric path for rows that genuinely had
no variable_types, on the grounds that it was a real choice rather than a
swallowed error. That was wrong, for a reason that only shows up once you look
at upstream: CRANE has no numeric grader at all. Its parse_answer
(gsm_symbolic.py:28-56) either runs validate_expression_equivalence or leaves
`correct = False`. Nothing in it ever compares two numbers.

So every example our numeric path graded was scored by a method CRANE does not
have, then averaged into a figure reported against CRANE's published number.
Keeping it "for rows we cannot grade symbolically" quietly answered a question
that should have stopped the run.

`answer_parsed` defaults to '' at dataset.py:65 and :164 -- falsy -- so this was
not a rare edge case: any source row lacking that field took the numeric path.

Both routes into it now raise, and the two helper methods it used
(`_extract_answer_gsm`, `_answers_match`) are deleted rather than left dormant.
"""

from __future__ import annotations

import pytest


class _FakeEvaluator:
    """Records which grader was used, so a silent switch is visible."""

    def __init__(self, symbolic_verdict=True, numeric_verdict=False):
        self.calls: list[str] = []
        self._symbolic_verdict = symbolic_verdict
        self._numeric_verdict = numeric_verdict
        self.symbolic_saw = None

    def _gsm_symbolic_equivalence(self, actual, expected, vt):
        self.calls.append("symbolic")
        self.symbolic_saw = vt
        return self._symbolic_verdict

    def _extract_answer_gsm(self, scored_output):
        self.calls.append("numeric_extract")
        return "42"

    def _answers_match(self, a, b):
        self.calls.append("numeric")
        return self._numeric_verdict


def _grade(evaluator, example):
    from synthesis.evaluate.benchmarks.gsm_symbolic.eval_logic import is_correct

    return is_correct(
        evaluator,
        actual="a * 2",
        expected="a + a",
        example=example,
        aux=None,
        scored_output="the answer is <<a * 2>>",
    )


def test_unreadable_variable_types_stops_instead_of_switching_graders():
    """The core rule: a field we could not read is not a grading decision."""
    evaluator = _FakeEvaluator()
    example = {
        "variable_types": "{'a': 'int'   <-- truncated",
        "answer_parsed": "a + a",
        "answer": "#### 42",
    }

    with pytest.raises(Exception) as caught:
        _grade(evaluator, example)

    assert not isinstance(caught.value, AssertionError), (
        "An unreadable variable_types was turned into an empty dict, which is "
        f"falsy, so grading silently fell through to {evaluator.calls}. The "
        "example was scored by a different method than the rest of the split "
        "and nothing recorded the switch."
    )


def test_variable_types_that_are_not_a_mapping_stop_too():
    evaluator = _FakeEvaluator()
    example = {
        "variable_types": "['a', 'b']",
        "answer_parsed": "a + a",
        "answer": "#### 42",
    }

    with pytest.raises(Exception) as caught:
        _grade(evaluator, example)

    assert not isinstance(caught.value, AssertionError)


def test_a_readable_variable_types_string_still_grades_symbolically():
    """Guard against over-fixing: the normal path must be untouched, and the
    grader must receive the parsed mapping."""
    evaluator = _FakeEvaluator(symbolic_verdict=True)
    example = {
        "variable_types": "{'a': 'int'}",
        "answer_parsed": "a + a",
        "answer": "#### 42",
    }

    assert _grade(evaluator, example) is True
    assert evaluator.calls == ["symbolic"]
    assert evaluator.symbolic_saw == {"a": "int"}


def test_a_variable_types_dict_still_grades_symbolically():
    evaluator = _FakeEvaluator(symbolic_verdict=True)
    example = {
        "variable_types": {"a": "int"},
        "answer_parsed": "a + a",
        "answer": "#### 42",
    }

    assert _grade(evaluator, example) is True
    assert evaluator.calls == ["symbolic"]


def test_a_row_without_variable_types_stops_instead_of_grading_numerically():
    """There is no second grader to fall back to any more.

    CRANE has exactly one way to grade GSM: prove the model's formula equals the
    gold's. Its parse_answer (gsm_symbolic.py:28-56) either runs
    validate_expression_equivalence or leaves correct = False -- there is no
    numeric comparison anywhere in it. Ours had one, and a row missing this
    field reached it silently.

    A row we cannot grade CRANE's way is a dataset problem. It must stop, not be
    scored by a method that measures something else and averaged in.
    """
    evaluator = _FakeEvaluator(numeric_verdict=True)
    example = {"answer_parsed": "a + a", "answer": "#### 42"}

    with pytest.raises(Exception) as caught:
        _grade(evaluator, example)

    assert not isinstance(caught.value, AssertionError), (
        f"Graded anyway, via {evaluator.calls}. A row with no variable_types "
        "was scored by numeric comparison and averaged in with examples that "
        "were graded by proof."
    )
    assert "numeric" not in evaluator.calls


def test_a_row_without_a_parsed_answer_stops_too():
    """`answer_parsed` defaults to '' at dataset.py:65 and :164, which is falsy,
    so a source row lacking the field routed straight to the numeric grader."""
    evaluator = _FakeEvaluator(numeric_verdict=True)
    example = {"variable_types": {"a": "int"}, "answer": "#### 42"}

    with pytest.raises(Exception) as caught:
        _grade(evaluator, example)

    assert not isinstance(caught.value, AssertionError)
    assert "numeric" not in evaluator.calls


def test_the_numeric_grader_is_gone_not_merely_unreachable():
    """A path left in place behind a branch nobody takes comes back.

    Both helpers existed only to serve the numeric comparison in is_correct --
    nothing else in synthesis/ called either one.

    The seven below are everything `_extract_answer_gsm` reached that nothing
    else reaches: the answer-expression puller, the expression evaluator, and
    the variable-parsing and substitution helpers underneath them. Found by
    grepping the whole repo for each name after each deletion, which uncovered
    two more (`_parse_symbolic_assignments`, `_safe_eval_arithmetic`) that only
    the first round's casualties called. Deliberately NOT in this list:
    `_truncate_gsm_output`, which has real callers left (evaluator.py:2565 and
    five sites in run_legacy_fixed_strategy.py).
    """
    from synthesis.evaluate.evaluator import Evaluator

    dead_grader = ("_extract_answer_gsm", "_answers_match")
    dead_subtree = (
        "_extract_answer_expression_gsm",
        "_evaluate_gsm_expression",
        "_resolve_symbolic_assignments",
        "_parse_variable_assignments",
        "_parse_symbolic_assignments",
        "_evaluate_symbolic_expression",
        "_safe_eval_arithmetic",
    )

    for dead in dead_grader + dead_subtree:
        assert not hasattr(Evaluator, dead), (
            f"Evaluator.{dead} still exists. It has no callers left, so it is "
            "the old grading method sitting dormant, ready to be wired back in."
        )


def test_the_helper_the_numeric_grader_shared_with_others_survives():
    """Guard against over-deleting. `_truncate_gsm_output` cuts the model's
    output at the point the answer should have appeared, and the syntax check
    still uses it. Removing it with the rest would break a path that has
    nothing to do with grading."""
    from synthesis.evaluate.evaluator import Evaluator

    assert hasattr(Evaluator, "_truncate_gsm_output")
