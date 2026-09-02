"""A broken harness must not be blamed on the strategy the AI wrote.

The behaviour this pins
-----------------------
When evaluation fails, feedback_loop.py does this (around line 2004):

    if not eval_result.success:
        attempt.failed_at = FailureStage.EVALUATION
        ...
        print("  Refining based on evaluation error...")
        -> sends the error text to the strategy-writing model
        -> asks it for a better strategy
        -> next attempt hits the same broken harness
        -> repeat

If the underlying error is `ModuleNotFoundError: no module named
'...vllm_startup'`, no strategy the model writes can possibly fix it. The loop
spends real API budget rewriting strategies against a missing file, and records
each round as "the strategy failed evaluation".

The distinguishing fact is already sitting in the result and nothing reads it:
`num_examples == 0`. A genuine 0% still evaluates its examples. If zero examples
ran, there is no measurement -- only a broken harness.

So: zero examples evaluated must be classified as a HARNESS failure, which stops
the run, and never as an EVALUATION failure, which asks the model to try again.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

# feedback_loop.py:22 imports ..generate.generator, which imports torch at module
# level, so this module cannot be imported on a machine without the GPU stack --
# even though the rule under test is pure Python.
#
# Two other test files already stub these the same way. Doing it here too, rather
# than once in a shared conftest, is deliberate: a global stub also satisfies
# `pytest.importorskip("torch")` in test_eos_requires_complete_prefix.py, which
# then runs those tests against a fake torch and reports five failures that are
# not real. An honest "skipped, needs torch" must not be turned into a
# misleading "failed" -- that is the same swap of a true signal for a
# plausible-looking wrong one that this whole sweep is about.
#
# Only stubs what is genuinely absent, so the GPU box uses the real library.
for _heavy in ("torch", "vllm", "transformers"):
    if _heavy not in sys.modules:
        try:
            __import__(_heavy)
        except ImportError:
            sys.modules[_heavy] = MagicMock()


@dataclass
class _EvalOutcome:
    """Stand-in for EvaluationResult carrying only what classification reads."""

    num_examples: int
    num_correct: int = 0
    accuracy: float = 0.0
    success: bool = False
    error: str | None = None


def test_there_is_a_separate_stage_for_harness_failure():
    """EVALUATION means 'the strategy scored badly'. That is a different thing
    from 'the evaluator could not run', and they must not share a name."""
    from synthesis.evaluate.feedback_loop import FailureStage

    assert hasattr(FailureStage, "HARNESS"), (
        "FailureStage has no HARNESS member, so a broken evaluator can only be "
        "recorded as EVALUATION -- indistinguishable from a strategy that "
        f"genuinely scored badly. Existing members: {[s.name for s in FailureStage]}"
    )
    assert FailureStage.HARNESS is not FailureStage.EVALUATION


def test_zero_examples_evaluated_is_a_harness_failure():
    """The core rule: no examples ran, so there is no score to explain."""
    from synthesis.evaluate.feedback_loop import FailureStage, classify_eval_failure

    outcome = _EvalOutcome(
        num_examples=0,
        success=False,
        error="No module named 'synthesis.evaluate.benchmarks.common.vllm_startup'",
    )

    assert classify_eval_failure(outcome) is FailureStage.HARNESS, (
        "An evaluation that ran zero examples was classified as a strategy "
        "problem. The pipeline will now ask the model to rewrite its strategy "
        "to fix a missing Python module, and will keep doing so every attempt."
    )


def test_a_real_zero_score_is_still_a_strategy_failure():
    """The opposite direction matters just as much.

    If examples really were evaluated and the strategy really scored 0, that IS
    feedback the model can act on. Misrouting it to HARNESS would stop the run
    on a legitimately bad strategy.
    """
    from synthesis.evaluate.feedback_loop import FailureStage, classify_eval_failure

    outcome = _EvalOutcome(
        num_examples=50, num_correct=0, accuracy=0.0, success=False,
        error="accuracy below threshold",
    )

    assert classify_eval_failure(outcome) is FailureStage.EVALUATION, (
        "A genuine 0% over 50 evaluated examples was treated as a broken "
        "harness. That would abort runs on strategies that are merely bad, "
        "which is exactly the feedback the loop exists to act on."
    )


@pytest.mark.parametrize("examples_run", [1, 7, 50])
def test_any_example_actually_evaluated_means_the_harness_worked(examples_run):
    from synthesis.evaluate.feedback_loop import FailureStage, classify_eval_failure

    outcome = _EvalOutcome(num_examples=examples_run, success=False)

    assert classify_eval_failure(outcome) is FailureStage.EVALUATION
