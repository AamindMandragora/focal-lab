"""A broken GSM grader must stop, not score every answer wrong.

The defect
----------
`Evaluator._gsm_symbolic_equivalence` ends like this:

    try:
        return bool(_crane_validate_expression_equivalence(...))
    except Exception:
        # Any hard failure in the proof path -> conservatively not-correct.
        return False

"Conservatively not-correct" sounds careful. It is not. The proof path imports
z3 inside the function, so on a machine where z3 is not installed the
ModuleNotFoundError lands in this catch and every single example is graded
wrong. Measured on this machine before the fix: `a * 2` against a gold of
`a + a` -- the same formula -- graded False.

That produces GSM accuracy = 0.0 with no indication that nothing was ever
proved. It is the same failure that cost weeks on Spider: a missing module
turned into a number, indistinguishable from a model that genuinely got
everything wrong.

The same applies just above it:

    try:
        variable_types = _ast.literal_eval(variable_types)
    except Exception:
        variable_types = {}

An empty type map means every variable is untyped, which sends the whole
expression down the not-provable path. A dataset field we could not read is
reported as a model that answered incorrectly.

Upstream CRANE has neither catch: `parse_answer` calls plain
`eval(batch[i]['variable_types'])` (gsm_symbolic.py:40) and lets a bad value
crash the run. So here strict CRANE parity and "never let a setup fault
masquerade as a measurement" happen to want the same thing.

What is deliberately NOT removed
--------------------------------
The early returns above these are real grading decisions, not swallowed
failures, and they stay:
  - model_expr is None            -> the model produced no answer
  - '**' in model_expr            -> CRANE's own guard (parse_answer:38)
  - pathological expression       -> our guard against hanging native code
Each of those is a genuine verdict about the model's output. The two catches
this file removes are verdicts about our own setup.
"""

from __future__ import annotations

import pytest


def _bare_evaluator():
    """An Evaluator with nothing wired up -- this method touches no state."""
    from synthesis.evaluate.evaluator import Evaluator

    return object.__new__(Evaluator)


# --------------------------------------------------------------------------
# the outer catch
# --------------------------------------------------------------------------


def test_a_failure_in_the_proof_path_stops_instead_of_scoring_wrong(monkeypatch):
    """The core rule, forced rather than depending on whether z3 is installed.

    Monkeypatching the prover keeps this test honest on both machines: it fails
    the same way on the GPU box, where z3 exists, as it does here.
    """
    import synthesis.evaluate.evaluator as evaluator_module

    def explode(*args, **kwargs):
        raise ModuleNotFoundError("No module named 'z3'")

    monkeypatch.setattr(
        evaluator_module, "_crane_validate_expression_equivalence", explode
    )

    with pytest.raises(ModuleNotFoundError, match="z3"):
        _bare_evaluator()._gsm_symbolic_equivalence("a * 2", "a + a", {"a": "int"})


def test_the_reason_the_prover_failed_survives(monkeypatch):
    """Replacing the cause with a generic error repeats the original mistake.
    Whatever broke must still be readable by whoever reads the traceback."""
    import synthesis.evaluate.evaluator as evaluator_module

    def explode(*args, **kwargs):
        raise RuntimeError("z3 solver aborted: out of memory")

    monkeypatch.setattr(
        evaluator_module, "_crane_validate_expression_equivalence", explode
    )

    with pytest.raises(RuntimeError, match="out of memory"):
        _bare_evaluator()._gsm_symbolic_equivalence("a * 2", "a + a", {"a": "int"})


def test_a_real_verdict_from_the_prover_is_passed_through(monkeypatch):
    """Guard against over-fixing: an actual answer must still be returned."""
    import synthesis.evaluate.evaluator as evaluator_module

    monkeypatch.setattr(
        evaluator_module,
        "_crane_validate_expression_equivalence",
        lambda *a, **k: True,
    )

    verdict = _bare_evaluator()._gsm_symbolic_equivalence("a * 2", "a + a", {"a": "int"})
    assert verdict is True


# --------------------------------------------------------------------------
# the variable_types catch
# --------------------------------------------------------------------------


def test_unreadable_variable_types_stops_instead_of_scoring_wrong():
    """A dataset field we could not parse is not evidence about the model."""
    with pytest.raises(Exception) as caught:
        _bare_evaluator()._gsm_symbolic_equivalence(
            "a * 2", "a + a", "{'a': 'int'   <-- truncated"
        )

    assert not isinstance(caught.value, AssertionError), (
        "Unreadable variable_types was swallowed and the answer scored wrong. "
        "Every example in the split would be graded against an empty type map, "
        "reporting 0% accuracy from a dataset problem."
    )


def test_variable_types_that_are_not_a_mapping_stop_too():
    """literal_eval succeeds here but yields a list, which is equally unusable."""
    with pytest.raises(Exception) as caught:
        _bare_evaluator()._gsm_symbolic_equivalence("a * 2", "a + a", "['a', 'b']")

    assert not isinstance(caught.value, AssertionError)


def test_a_readable_variable_types_string_is_still_parsed(monkeypatch):
    """Guard against over-fixing: the normal string-to-dict path must survive,
    and the prover must receive the parsed mapping."""
    import synthesis.evaluate.evaluator as evaluator_module

    seen = {}

    def record(model_expr, gold_expr, var_types):
        seen["var_types"] = var_types
        return True

    monkeypatch.setattr(
        evaluator_module, "_crane_validate_expression_equivalence", record
    )

    verdict = _bare_evaluator()._gsm_symbolic_equivalence(
        "a * 2", "a + a", "{'a': 'int'}"
    )

    assert verdict is True
    assert seen["var_types"] == {"a": "int"}


# --------------------------------------------------------------------------
# the real grading decisions above must be untouched
# --------------------------------------------------------------------------


def test_no_answer_from_the_model_is_still_wrong_not_an_error():
    assert _bare_evaluator()._gsm_symbolic_equivalence(None, "a + a", {"a": "int"}) is False


def test_the_crane_double_star_guard_is_still_wrong_not_an_error():
    """CRANE's own guard (parse_answer:38) -- a verdict about the model."""
    assert (
        _bare_evaluator()._gsm_symbolic_equivalence("a ** 2", "a + a", {"a": "int"})
        is False
    )


def test_a_runaway_expression_is_still_wrong_not_an_error():
    """Our anti-hang guard -- also a verdict about the model's output."""
    runaway = " + ".join(["a"] * 500)
    assert (
        _bare_evaluator()._gsm_symbolic_equivalence(runaway, "a + a", {"a": "int"})
        is False
    )
