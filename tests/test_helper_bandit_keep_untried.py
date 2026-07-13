"""The helper policy must not remove helpers that have never been tried."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


for _heavy in ("torch", "vllm", "transformers"):
    if _heavy not in sys.modules:
        try:
            __import__(_heavy)
        except Exception:
            sys.modules[_heavy] = MagicMock()

from synthesis.evaluate.feedback_loop import SynthesisAttempt, SynthesisPipeline  # noqa: E402


def _make_pipeline(universe):
    pipe = SynthesisPipeline.__new__(SynthesisPipeline)
    pipe._helper_universe = set(universe)
    pipe.helper_bandit_min_evals = 3
    pipe.helper_bandit_top_k = 12
    pipe.helper_bandit_ucb_c = 0.35
    pipe.helper_bandit_explore_untried = 1
    pipe.require_delimiters = False
    pipe.eval_max_seconds_per_example = None
    pipe.min_accuracy = 0.0
    pipe.min_syntax_rate = 0.0
    return pipe


def _attempt(number, helpers_used, accuracy, syntax_rate):
    body = "\n".join(f"var _ := helpers.{name}(lm, parser);" for name in helpers_used)
    return SynthesisAttempt(
        attempt_number=number,
        strategy_code=body,
        full_dafny_code=body,
        timestamp="2026-06-19T00:00:00",
        eval_result=SimpleNamespace(
            accuracy=accuracy,
            syntax_rate=syntax_rate,
            contains_delimiters=True,
            max_sample_time_seconds=1.0,
            num_examples=10,
        ),
    )


UNIVERSE = sorted(SynthesisPipeline.PRUNABLE_HELPERS)
BEST_HELPER = "AdaptiveConstrainedStep"
OTHER_TRIED = ["ConstrainedGeneration", "UnconstrainedChunk"]


def _attempts():
    return [
        _attempt(1, [BEST_HELPER], 0.44, 0.81),
        _attempt(2, [BEST_HELPER] + OTHER_TRIED, 0.05, 0.10),
        _attempt(3, OTHER_TRIED, 0.02, 0.07),
        _attempt(4, [BEST_HELPER], 0.40, 0.80),
    ]


def test_all_untried_helpers_stay_on_the_menu():
    pipe = _make_pipeline(UNIVERSE)
    allowed, status = pipe._compute_allowed_helpers_bandit(_attempts())
    tried = {BEST_HELPER, *OTHER_TRIED}
    missing = [name for name in UNIVERSE if name not in tried and name not in set(allowed)]
    assert not missing, f"untried helpers were removed: {missing}; status={status}"


def test_mask_is_still_active():
    pipe = _make_pipeline(UNIVERSE)
    allowed, _ = pipe._compute_allowed_helpers_bandit(_attempts())
    prunable_allowed = set(allowed) & SynthesisPipeline.PRUNABLE_HELPERS
    assert len(prunable_allowed) < len(SynthesisPipeline.PRUNABLE_HELPERS)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
