"""REx outer-bandit arm selection (Thompson sampling over Beta posteriors)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from synthesis.evaluate.search_tree import SearchNode


class RexArm(Protocol):
    node_id: int
    heuristic_h: float
    fail_count: int


def beta_posterior_params(h: float, fail_count: int, temperature: float) -> tuple[float, float]:
    """Return (alpha, beta) for REx Eq. 10."""
    c = max(0.0, temperature)
    h_clamped = min(1.0, max(0.0, h))
    alpha = 1.0 + c * h_clamped
    beta = 1.0 + c * (1.0 - h_clamped) + max(0, fail_count)
    return alpha, beta


class RexBandit:
    """Select tree nodes to refine using REx Thompson sampling."""

    def __init__(self, temperature: float = 2.0, rng: np.random.Generator | None = None):
        self.temperature = max(0.0, temperature)
        self._rng = rng or np.random.default_rng()

    def select_arm(self, nodes: list[SearchNode]) -> SearchNode:
        if not nodes:
            raise ValueError("REx select_arm requires at least one node")
        if len(nodes) == 1:
            return nodes[0]

        best_node = nodes[0]
        best_sample = -1.0
        for node in nodes:
            alpha, beta = beta_posterior_params(
                node.heuristic_h, node.fail_count, self.temperature
            )
            sample = float(self._rng.beta(alpha, beta))
            if sample > best_sample:
                best_sample = sample
                best_node = node
        return best_node

    def record_pull(self, parent: SearchNode, child_met_threshold: bool) -> None:
        if not child_met_threshold:
            parent.fail_count += 1
