"""Our GSM grader must match CRANE's, including where CRANE's is wrong.

The decision this encodes
-------------------------
GSM accuracy is reported against CRANE's published numbers. A grader that is
*better* than CRANE's produces numbers that cannot be compared to theirs, which
is worse for the paper than a grader that is faithfully wrong. So where our port
deviates from upstream, upstream wins -- even where upstream is plainly buggy.

Upstream is `src/prompting/gsm_symbolic.py` in github.com/uiuc-focal-lab/CRANE:
`test_expression_equivalence` (line 96) and `validate_expression_equivalence`
(line 131). Our ports are `_crane_test_expression_equivalence` and
`_crane_validate_expression_equivalence` in synthesis/evaluate/evaluator.py.

The two deviations this file pins
---------------------------------
1. `round(` in the gold expression.

   Upstream line 168:  re.sub(r'\\round\\(', 'ToInt(', expr2)

   That `\\r` is the regex escape for a carriage return, so the pattern is
   CR + "ound(" and never matches anything. Upstream's round->ToInt conversion
   is dead code. The gold keeps its `round(`, `safe_eval` then throws because the
   eval environment has no builtins, and grading falls through to the random
   sampler -- where a plain `eval` does have `round` and computes it correctly.

   Our port "fixed" the typo. That is not a fix: ToInt truncates and round
   rounds half away from zero. int(0.667) is 0, round(0.667) is 1. Converting
   both `int(` and `round(` to `ToInt(` makes them the same function, so a model
   that answers int(x/3) is proved equivalent to a gold of round(x/3) and scored
   correct. That inflates accuracy relative to CRANE.

2. An untyped variable inside the sampler.

   Upstream never assigns a value to a variable whose type is not one of the
   three it knows, so the variable survives into the substituted string and the
   `eval` raises NameError. Which answer that produces depends on which
   expression the variable is in -- and upstream's two catches are asymmetric:
   the model's formula throwing means wrong, the gold's formula throwing means
   correct. So an untyped variable appearing only in the gold scores the model
   CORRECT.

   Our port short-circuits to False before sampling. Stricter than upstream, so
   it deflates accuracy relative to CRANE.

Why z3 is not stubbed here
--------------------------
z3 is not installed on this machine. The behavioural test below is skipped
rather than run against a fake solver. A MagicMock z3 would answer every query
with a truthy Mock, so the test would report a result it never actually
computed -- which is the same swap of a real signal for a plausible-looking
fake one that this whole sweep is about. Skipped is the honest answer here; the
test runs for real on the GPU box, where z3 exists.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

EVALUATOR_SOURCE = (
    Path(__file__).resolve().parents[1] / "synthesis" / "evaluate" / "evaluator.py"
)


# --------------------------------------------------------------------------
# 1. the round( substitution
# --------------------------------------------------------------------------


def check_gold_round_is_left_alone(source: str) -> tuple[bool, str]:
    """Does the substitution applied to the gold expression leave `round(` alone?

    A plain function, not a fixture, so the test below can run it against the
    known-bad shape and confirm it says no. A structural check that cannot fail
    is not a check.
    """
    applied = re.search(
        r"""expr2\s*=\s*re\.sub\(\s*(r?["'][^"']*ound[^"']*["'])\s*,""",
        source,
    )
    if applied is None:
        return False, (
            "Could not find the re.sub that rewrites round( in the gold "
            "expression. If it was deleted outright that is also a deviation: "
            "upstream still runs the substitution, it just never matches."
        )

    pattern = ast.literal_eval(applied.group(1))
    rewritten = re.sub(pattern, "ToInt(", "round(x/3)")
    if rewritten != "round(x/3)":
        return False, (
            f"The pattern {applied.group(1)} rewrote 'round(x/3)' to "
            f"'{rewritten}'. Upstream's pattern never matches, so the gold keeps "
            "its round( and grading falls through to the sampler. Converting it "
            "to ToInt( makes round and int the same function and scores wrong "
            "model answers as correct."
        )
    return True, "round( in the gold expression is left alone, as upstream leaves it"


def test_the_gold_expression_keeps_its_round_call():
    ok, detail = check_gold_round_is_left_alone(EVALUATOR_SOURCE.read_text())
    assert ok, detail


# The shape that was in the file before this fix -- the typo "repaired". If the
# check above passes against this, the check is worthless and proves nothing.
REPAIRED_REGEX_SHAPE = '''
    if "round(" in expr1:
        return False
    expr2 = re.sub(r"round\\(", "ToInt(", expr2)
'''


def test_the_check_rejects_the_repaired_regex():
    ok, detail = check_gold_round_is_left_alone(REPAIRED_REGEX_SHAPE)
    assert not ok, (
        "The check passed against the live-regex shape, so it would pass no "
        "matter what the file says and proves nothing."
    )
    assert "ToInt(x/3)" in detail


def test_int_and_round_are_not_proved_to_be_the_same_function():
    """The score-inflating consequence, measured end to end.

    A model answering int(x/3) against a gold of round(x/3) is wrong -- they
    disagree whenever the fraction is at or above a half. CRANE grades this by
    sampling and marks it wrong. If we rewrite both to ToInt( the solver proves
    them identical and marks it correct.
    """
    pytest.importorskip(
        "z3",
        reason="needs the real solver; a stubbed z3 would fake the answer",
    )
    from synthesis.evaluate.evaluator import _crane_validate_expression_equivalence

    verdict = _crane_validate_expression_equivalence(
        "int(x/3)", "round(x/3)", {"x": "int"}
    )

    assert verdict is False, (
        "int(x/3) was accepted as equivalent to round(x/3). Both were rewritten "
        "to ToInt(, so the solver compared the gold against itself. Every model "
        "answer that truncates where the gold rounds is now scored correct."
    )


# --------------------------------------------------------------------------
# 2. the untyped variable in the sampler
# --------------------------------------------------------------------------


def test_a_variable_missing_from_the_types_dict_raises_like_upstream():
    """Upstream indexes the dict directly: `var_types[var]` (line 101).

    A variable name the identifier regex picked up but that has no entry at all
    raises KeyError and stops the run. Softening this to `.get()` invents a
    verdict upstream never produces -- which direction it lands in depends on
    which expression the variable sits in, so it can push the score either way.
    """
    from synthesis.evaluate.evaluator import _crane_test_expression_equivalence

    with pytest.raises(KeyError):
        _crane_test_expression_equivalence("a * 2", "a * b", {"a", "b"}, {"a": "int"})


def test_sampler_scores_correct_when_only_the_gold_formula_throws():
    """Upstream's asymmetry, reproduced exactly.

    This is the case where `b` IS in the dict but carries a type upstream does
    not recognise, so none of its three branches fire and `b` is never assigned
    a value. It survives into the substituted string. The model's formula does
    not mention it and evaluates fine; the gold's does and raises NameError,
    which upstream's second catch turns into `return True`.

    Indefensible as grading, and it is what CRANE does, so it is what we do.
    Our port used to refuse to sample at all here, which returned False and
    undercounted relative to CRANE.
    """
    from synthesis.evaluate.evaluator import _crane_test_expression_equivalence

    verdict = _crane_test_expression_equivalence(
        "a * 2", "a * b", {"a", "b"}, {"a": "int", "b": "str"}
    )

    assert verdict is True, (
        "Returned False where CRANE returns True. CRANE leaves a variable with "
        "an unrecognised type unsubstituted, so the gold expression raises and "
        "its catch returns True. Refusing to sample is stricter than CRANE and "
        "makes our GSM accuracy not comparable to their published number."
    )


def test_sampler_scores_wrong_when_the_model_formula_throws():
    """The other side of the asymmetry. Guards against over-correcting: this
    direction already matches upstream and must stay put."""
    from synthesis.evaluate.evaluator import _crane_test_expression_equivalence

    verdict = _crane_test_expression_equivalence(
        "a * b", "a * 2", {"a", "b"}, {"a": "int", "b": "str"}
    )

    assert verdict is False, (
        "A variable with an unrecognised type in the model's own formula must "
        "score wrong -- upstream's first catch returns False."
    )


def test_sampler_still_agrees_on_formulas_that_are_genuinely_equal():
    """Normal grading must be untouched by either change."""
    from synthesis.evaluate.evaluator import _crane_test_expression_equivalence

    assert (
        _crane_test_expression_equivalence(
            "a * 2", "a + a", {"a"}, {"a": "int"}
        )
        is True
    )


def test_sampler_still_separates_formulas_that_genuinely_differ():
    from synthesis.evaluate.evaluator import _crane_test_expression_equivalence

    assert (
        _crane_test_expression_equivalence(
            "a * 2", "a * 3", {"a"}, {"a": "int"}
        )
        is False
    )
