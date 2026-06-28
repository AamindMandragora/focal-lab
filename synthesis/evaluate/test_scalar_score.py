"""Tests for SynthesisPipeline._evaluation_scalar_score (A1 fix).

The scalar drives the bandit's best-attempt selection and helper-credit means.
The old formula (accuracy + syntax_rate + binary delimiter + binary runtime)
let a single over-budget example sink a high-accuracy attempt below a
low-accuracy-but-fast attempt. The fix weights accuracy 3x and grades runtime
as the fraction of examples within budget instead of a binary 0/1.

Called as an unbound method on a duck-typed stand-in, since the method only
reads self.require_delimiters and self.eval_max_seconds_per_example.
"""
from types import SimpleNamespace

from synthesis.evaluate.feedback_loop import SynthesisPipeline


def _score(result, *, require_delimiters=False, budget=300.0):
    fake_self = SimpleNamespace(
        require_delimiters=require_delimiters,
        eval_max_seconds_per_example=budget,
    )
    return SynthesisPipeline._evaluation_scalar_score(fake_self, result)


def _result(accuracy, syntax_rate, n, n_over_budget, max_t):
    samples = [{"runtime_budget_exceeded": i < n_over_budget} for i in range(n)]
    return SimpleNamespace(
        accuracy=accuracy,
        syntax_rate=syntax_rate,
        contains_delimiters=False,
        max_sample_time_seconds=max_t,
        sample_outputs=samples,
    )


def test_high_accuracy_one_timeout_beats_low_accuracy_fast():
    # Reproduces the observed att2 vs att4 inversion.
    att2 = _result(0.453, 0.793, 300, 1, 350.0)   # high acc, one example over 300s
    att4 = _result(0.270, 0.477, 300, 0, 120.0)   # low acc, all fast
    assert _score(att2) > _score(att4), (
        f"att2={_score(att2):.3f} must beat att4={_score(att4):.3f}; "
        "one slow example must not sink a higher-accuracy attempt"
    )


def test_accuracy_dominates_when_other_terms_equal():
    hi = _result(0.60, 0.90, 100, 0, 50.0)
    lo = _result(0.40, 0.90, 100, 0, 50.0)
    assert _score(hi) > _score(lo)
    # 0.2 accuracy gap -> 0.6 scalar (3x), strictly larger than any single binary term (0.25).
    assert _score(hi) - _score(lo) >= 0.5


def test_runtime_is_graded_not_binary():
    few = _result(0.50, 0.80, 100, 1, 400.0)
    many = _result(0.50, 0.80, 100, 50, 400.0)
    assert _score(few) > _score(many), "more over-budget examples must score strictly lower"


def test_no_budget_gives_full_runtime_credit():
    r = _result(0.5, 0.5, 10, 0, 0.0)
    assert abs(_score(r, budget=None) - (3 * 0.5 + 0.5 + 0.25 + 0.25)) < 1e-9


def test_empty_samples_default_full_credit():
    r = SimpleNamespace(
        accuracy=0.5, syntax_rate=0.5, contains_delimiters=False,
        max_sample_time_seconds=0.0, sample_outputs=[],
    )
    assert abs(_score(r, budget=300.0) - (3 * 0.5 + 0.5 + 0.25 + 0.25)) < 1e-9


if __name__ == "__main__":
    test_high_accuracy_one_timeout_beats_low_accuracy_fast()
    test_accuracy_dominates_when_other_terms_equal()
    test_runtime_is_graded_not_binary()
    test_no_budget_gives_full_runtime_credit()
    test_empty_samples_default_full_credit()
    print("All tests passed.")
