"""Tests for SynthesisPipeline._compute_prunable_helper_marginals (A2 fix).

Old behavior: the full composite scalar of an attempt was appended to EVERY
prunable helper it used, so a helper that drove high accuracy but co-occurred
with a slow/bad helper had its credit dragged down by a penalty it did not
cause. The fix credits each helper by its counterfactual marginal:
mean(scalar | helper used) - mean(scalar | helper NOT used), isolating the
helper's own contribution. Helpers used in every attempt (no "without" set)
are reported as ubiquitous so the bandit protects rather than prunes them.

Methods are called unbound on a duck-typed stand-in; the credit method only
reads self.PRUNABLE_HELPERS, self._helper_universe, self.require_delimiters,
self.eval_max_seconds_per_example, and self._get_helper_calls_for_evaluation_history.
"""
from types import SimpleNamespace

from synthesis.evaluate.feedback_loop import SynthesisPipeline


def _pipe(helper_map):
    fake = SimpleNamespace(
        PRUNABLE_HELPERS={"Good", "Bad", "Ubi", "NeverUsed"},
        _helper_universe={"Good", "Bad", "Ubi", "NeverUsed"},
        require_delimiters=False,
        eval_max_seconds_per_example=None,  # runtime_frac -> 1.0
    )
    fake._get_helper_calls_for_evaluation_history = lambda code: helper_map[code]
    # Bind the real scalar method onto the stand-in (the credit method calls it).
    fake._evaluation_scalar_score = lambda result: SynthesisPipeline._evaluation_scalar_score(fake, result)
    return fake


def _att(code, accuracy, syntax_rate):
    return SimpleNamespace(
        strategy_code=code,
        eval_result=SimpleNamespace(
            accuracy=accuracy, syntax_rate=syntax_rate,
            contains_delimiters=False, max_sample_time_seconds=0.0, sample_outputs=[],
        ),
    )


def _marginals(pipe, attempts):
    return SynthesisPipeline._compute_prunable_helper_marginals(pipe, attempts)


def _scalar(pipe, att):
    return SynthesisPipeline._evaluation_scalar_score(pipe, att.eval_result)


def test_marginal_credit_isolates_helper_contribution():
    helper_map = {
        "a1": {"Good"}, "a2": {"Good"},       # Good in high-score attempts
        "a3": {"Bad"}, "a4": {"Bad"},         # Bad in low-score attempts
        "a5": {"Good", "Bad"},                # co-occurrence (low score)
    }
    pipe = _pipe(helper_map)
    attempts = [
        _att("a1", 0.9, 1.0), _att("a2", 0.9, 1.0),
        _att("a3", 0.1, 0.5), _att("a4", 0.1, 0.5),
        _att("a5", 0.1, 0.5),
    ]
    m = _marginals(pipe, attempts)

    scalars = {a.strategy_code: _scalar(pipe, a) for a in attempts}

    def expected(helper):
        with_s = [scalars[c] for c, hs in helper_map.items() if helper in hs]
        without_s = [scalars[c] for c, hs in helper_map.items() if helper not in hs]
        return sum(with_s) / len(with_s) - sum(without_s) / len(without_s)

    assert m["Good"][0] > 0, "Good drove high scores -> positive marginal"
    assert m["Bad"][0] < 0, "Bad only in low scores -> negative marginal"
    # Co-occurrence with Bad must NOT flip Good's credit to negative.
    assert abs(m["Good"][0] - expected("Good")) < 1e-9
    assert abs(m["Bad"][0] - expected("Bad")) < 1e-9
    # n_with pull counts
    assert m["Good"][1] == 3
    assert m["Bad"][1] == 3


def test_ubiquitous_helper_is_flagged_and_protected():
    # Ubi appears in every attempt -> no "without" set -> marginal undefined.
    helper_map = {"b1": {"Ubi"}, "b2": {"Ubi", "Good"}, "b3": {"Ubi", "Bad"}}
    pipe = _pipe(helper_map)
    attempts = [_att("b1", 0.5, 0.5), _att("b2", 0.9, 1.0), _att("b3", 0.1, 0.5)]
    m = _marginals(pipe, attempts)
    assert m["Ubi"][2] is True, "helper used in every attempt must be flagged ubiquitous"
    assert m["Ubi"][0] == 0.0, "ubiquitous helper gets neutral (0.0) marginal, not a drag"


def test_untried_helper_absent_from_marginals():
    helper_map = {"c1": {"Good"}}
    pipe = _pipe(helper_map)
    m = _marginals(pipe, [_att("c1", 0.5, 0.5)])
    assert "NeverUsed" not in m, "untried helpers are handled by the bandit, not credited here"


if __name__ == "__main__":
    test_marginal_credit_isolates_helper_contribution()
    test_ubiquitous_helper_is_flagged_and_protected()
    test_untried_helper_absent_from_marginals()
    print("All tests passed.")
