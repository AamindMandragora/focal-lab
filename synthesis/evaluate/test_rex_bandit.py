"""Unit tests for REx outer-bandit selection."""

import numpy as np

from synthesis.evaluate.rex_bandit import RexBandit, beta_posterior_params
from synthesis.evaluate.search_tree import SearchNode


def _node(node_id: int, goodness: float, fail_count: int = 0) -> SearchNode:
    h = min(1.0, max(0.0, goodness))
    return SearchNode(
        node_id=node_id,
        parent_id=None if node_id == 0 else 0,
        attempt_number=node_id + 1,
        strategy_code=f"strategy_{node_id}",
        full_dafny_code=f"full_{node_id}",
        timestamp="t",
        goodness=goodness,
        heuristic_h=h,
        fail_count=fail_count,
    )


def test_beta_posterior_params_match_rex_formula():
    alpha, beta = beta_posterior_params(h=0.6, fail_count=3, temperature=2.0)
    assert alpha == 1.0 + 2.0 * 0.6
    assert beta == 1.0 + 2.0 * (1.0 - 0.6) + 3.0


def test_record_pull_increments_fail_count_only_on_miss():
    parent = _node(0, goodness=0.2)
    rex = RexBandit(temperature=2.0)
    rex.record_pull(parent, child_met_threshold=False)
    assert parent.fail_count == 1
    rex.record_pull(parent, child_met_threshold=True)
    assert parent.fail_count == 1


def test_high_goodness_arm_selected_more_often():
    low = _node(0, goodness=0.1)
    high = _node(1, goodness=0.9)
    rng = np.random.default_rng(0)
    rex = RexBandit(temperature=2.0, rng=rng)
    picks = [rex.select_arm([low, high]).node_id for _ in range(200)]
    assert picks.count(1) > picks.count(0)
