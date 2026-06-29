"""Unit tests for goodness scoring."""

from synthesis.evaluate.evaluator import EvaluationResult
from synthesis.evaluate.feedback_loop import FailureStage, SynthesisAttempt
from synthesis.evaluate.goodness import (
    compute_goodness_from_attempt,
    evaluation_scalar_score,
    heuristic_h,
    scalar_target,
)


def _eval_result(
    *,
    accuracy: float,
    syntax_rate: float,
    contains_delimiters: bool = True,
    num_examples: int = 10,
) -> EvaluationResult:
    return EvaluationResult(
        success=True,
        accuracy=accuracy,
        contains_delimiters=contains_delimiters,
        syntax_rate=syntax_rate,
        num_examples=num_examples,
        num_correct=int(accuracy * num_examples),
        total_time_seconds=1.0,
    )


def test_goodness_zero_on_verification_failure():
    attempt = SynthesisAttempt(
        attempt_number=1,
        strategy_code="method body",
        full_dafny_code="full",
        timestamp="t",
        failed_at=FailureStage.VERIFICATION,
    )
    goodness = compute_goodness_from_attempt(
        attempt,
        min_accuracy=0.4,
        min_syntax_rate=0.9,
        require_delimiters=True,
        eval_max_seconds_per_example=90.0,
    )
    assert goodness == 0.0


def test_goodness_zero_on_zero_examples():
    attempt = SynthesisAttempt(
        attempt_number=1,
        strategy_code="method body",
        full_dafny_code="full",
        timestamp="t",
        eval_result=_eval_result(accuracy=0.0, syntax_rate=0.0, num_examples=0),
    )
    goodness = compute_goodness_from_attempt(
        attempt,
        min_accuracy=0.4,
        min_syntax_rate=0.9,
        require_delimiters=True,
        eval_max_seconds_per_example=90.0,
    )
    assert goodness == 0.0


def test_goodness_near_one_at_threshold_perfect():
    target = scalar_target(
        min_accuracy=0.4,
        min_syntax_rate=0.9,
        require_delimiters=True,
        eval_max_seconds_per_example=90.0,
    )
    result = _eval_result(accuracy=0.4, syntax_rate=0.9)
    score = evaluation_scalar_score(
        result,
        require_delimiters=True,
        eval_max_seconds_per_example=90.0,
    )
    assert abs(score - target) < 1e-9

    attempt = SynthesisAttempt(
        attempt_number=1,
        strategy_code="method body",
        full_dafny_code="full",
        timestamp="t",
        eval_result=result,
    )
    goodness = compute_goodness_from_attempt(
        attempt,
        min_accuracy=0.4,
        min_syntax_rate=0.9,
        require_delimiters=True,
        eval_max_seconds_per_example=90.0,
    )
    assert abs(goodness - 1.0) < 1e-9


def test_goodness_exceeds_one_when_accuracy_overshoots():
    attempt = SynthesisAttempt(
        attempt_number=1,
        strategy_code="method body",
        full_dafny_code="full",
        timestamp="t",
        eval_result=_eval_result(accuracy=0.8, syntax_rate=0.9),
    )
    goodness = compute_goodness_from_attempt(
        attempt,
        min_accuracy=0.4,
        min_syntax_rate=0.9,
        require_delimiters=True,
        eval_max_seconds_per_example=90.0,
    )
    assert goodness > 1.0


def test_heuristic_h_clamps():
    assert heuristic_h(0.0) == 0.0
    assert heuristic_h(0.5) == 0.5
    assert heuristic_h(1.0) == 1.0
    assert heuristic_h(1.5) == 1.0
