"""Bandit helper-mask policy: untried arms must NOT be pruned.

User ruling 2026-06-19: an untried helper (zero pulls) has no evidence against
it, so the adaptive mask must keep ALL untried helpers on the author's menu each
iteration -- not just the single alphabetically-first one. The mask should only
ever prune helpers that HAVE been tried and did worse, never untried ones.

These tests drive `_compute_allowed_helpers_bandit` directly with a controlled
helper universe and a small set of evaluated attempts that try only a couple of
helpers (the realistic low-adoption case).
"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# The bandit logic under test is pure Python, but importing the synthesis package
# pulls in heavy ML deps (torch etc.) that aren't installed on the dev Mac. Stub
# any that are missing so the import succeeds without a GPU stack.
for _heavy in ("torch", "vllm", "transformers"):
    if _heavy not in sys.modules:
        try:
            __import__(_heavy)
        except Exception:
            sys.modules[_heavy] = MagicMock()

from synthesis.evaluate.feedback_loop import SynthesisPipeline, SynthesisAttempt


def _make_pipeline(universe):
    """Build a SynthesisPipeline shell with only the bandit-relevant attributes set."""
    pipe = SynthesisPipeline.__new__(SynthesisPipeline)
    pipe._helper_universe = set(universe)
    pipe.helper_bandit_min_evals = 3
    pipe.helper_bandit_top_k = 12
    pipe.helper_bandit_ucb_c = 0.35
    pipe.helper_bandit_explore_untried = 1
    pipe.require_delimiters = False
    pipe.eval_max_seconds_per_example = None
    return pipe


def _eval_result(accuracy, syntax_rate):
    return SimpleNamespace(
        accuracy=accuracy,
        syntax_rate=syntax_rate,
        contains_delimiters=True,
        max_sample_time_seconds=1.0,
    )


def _attempt(number, helpers_used, accuracy, syntax_rate):
    body = "\n".join(f"var _ := helpers.{name}(lm, parser);" for name in helpers_used)
    return SynthesisAttempt(
        attempt_number=number,
        strategy_code=body,
        full_dafny_code=body,
        timestamp="2026-06-19T00:00:00",
        eval_result=_eval_result(accuracy, syntax_rate),
    )


# Universe = the full set of prunable helpers (matches a real run where the whole
# tool menu is prunable). The author tries only a handful.
UNIVERSE = sorted(SynthesisPipeline.PRUNABLE_HELPERS)

# Helpers the author actually exercised across attempts (low adoption, realistic).
BEST_HELPER = "AdaptiveConstrainedStep"
OTHER_TRIED = ["ConstrainedGeneration", "UnconstrainedChunk"]


def _attempts():
    return [
        _attempt(1, [BEST_HELPER], accuracy=0.44, syntax_rate=0.81),          # best
        _attempt(2, [BEST_HELPER] + OTHER_TRIED, accuracy=0.05, syntax_rate=0.10),
        _attempt(3, OTHER_TRIED, accuracy=0.02, syntax_rate=0.07),
        _attempt(4, [BEST_HELPER], accuracy=0.40, syntax_rate=0.80),
    ]


def test_all_untried_helpers_stay_on_the_menu():
    """Every prunable helper with zero pulls must remain in the allowed set."""
    pipe = _make_pipeline(UNIVERSE)
    attempts = _attempts()

    allowed, status = pipe._compute_allowed_helpers_bandit(attempts)
    allowed = set(allowed)

    tried = {BEST_HELPER, *OTHER_TRIED}
    untried = [h for h in UNIVERSE if h not in tried]

    missing = [h for h in untried if h not in allowed]
    assert not missing, (
        f"{len(missing)} untried helper(s) were pruned but should be kept: "
        f"{missing[:10]}{'...' if len(missing) > 10 else ''} | status={status}"
    )


def test_mask_is_still_active():
    """Keeping untried arms must not turn the mask off: tried-but-not-best
    helpers (real evidence they did worse) can still be pruned."""
    pipe = _make_pipeline(UNIVERSE)
    attempts = _attempts()

    allowed, _ = pipe._compute_allowed_helpers_bandit(attempts)
    prunable_allowed = set(allowed) & SynthesisPipeline.PRUNABLE_HELPERS

    # At least one prunable helper is disabled -> the mask still does something.
    assert len(prunable_allowed) < len(SynthesisPipeline.PRUNABLE_HELPERS)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
