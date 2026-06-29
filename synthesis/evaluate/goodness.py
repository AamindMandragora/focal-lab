"""Goodness scoring for REx search-tree synthesis."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from synthesis.evaluate.evaluator import EvaluationResult
    from synthesis.evaluate.feedback_loop import FailureStage, SynthesisAttempt

_SCALAR_EPS = 1e-9


def scalar_target(
    *,
    min_accuracy: float,
    min_syntax_rate: float,
    require_delimiters: bool,
    eval_max_seconds_per_example: Optional[float],
) -> float:
    """Reference scalar at configured thresholds with perfect secondary terms."""
    delimiter_term = 0.25 if require_delimiters else 0.0
    runtime_term = 0.25 if eval_max_seconds_per_example is not None else 0.0
    return (
        3.0 * min_accuracy
        + min_syntax_rate
        + delimiter_term
        + runtime_term
    )


def evaluation_scalar_score(
    result: EvaluationResult,
    *,
    require_delimiters: bool,
    eval_max_seconds_per_example: Optional[float],
) -> float:
    """Scalar score shared by helper-bandit marginals and goodness."""
    delimiter_score = 1.0 if (result.contains_delimiters or not require_delimiters) else 0.0

    if eval_max_seconds_per_example is None:
        runtime_frac = 1.0
    else:
        samples = getattr(result, "sample_outputs", None) or []
        if not samples:
            runtime_frac = 1.0
        else:
            within = sum(
                1 for s in samples if not s.get("runtime_budget_exceeded", False)
            )
            runtime_frac = within / len(samples)

    return (
        3.0 * result.accuracy
        + result.syntax_rate
        + 0.25 * delimiter_score
        + 0.25 * runtime_frac
    )


def heuristic_h(goodness: float) -> float:
    """Map goodness to REx prior center in [0, 1]."""
    if goodness <= 0.0:
        return 0.0
    return min(1.0, goodness)


def compute_goodness_from_attempt(
    attempt: SynthesisAttempt,
    *,
    min_accuracy: float,
    min_syntax_rate: float,
    require_delimiters: bool,
    eval_max_seconds_per_example: Optional[float],
) -> float:
    """Return nonnegative goodness for a synthesis attempt."""
    from synthesis.evaluate.feedback_loop import FailureStage

    if not (attempt.strategy_code or "").strip():
        return 0.0

    if attempt.failed_at in {
        FailureStage.SEARCH_CONTRACT,
        FailureStage.VERIFICATION,
        FailureStage.COMPILATION,
        FailureStage.RUNTIME,
    }:
        return 0.0

    if attempt.eval_result is None:
        return 0.0

    if (attempt.eval_result.num_examples or 0) == 0:
        return 0.0

    target = scalar_target(
        min_accuracy=min_accuracy,
        min_syntax_rate=min_syntax_rate,
        require_delimiters=require_delimiters,
        eval_max_seconds_per_example=eval_max_seconds_per_example,
    )
    score = evaluation_scalar_score(
        attempt.eval_result,
        require_delimiters=require_delimiters,
        eval_max_seconds_per_example=eval_max_seconds_per_example,
    )
    return score / max(target, _SCALAR_EPS)
