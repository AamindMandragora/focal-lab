"""Explicit immutable search tree for REx synthesis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from synthesis.evaluate.feedback_loop import FailureStage
    from synthesis.evaluate.evaluator import EvaluationResult

from synthesis.evaluate.goodness import heuristic_h


@dataclass
class SearchNode:
    """Immutable snapshot of one CSD candidate in the search tree."""

    node_id: int
    parent_id: Optional[int]
    attempt_number: int
    strategy_code: str
    full_dafny_code: str
    timestamp: str
    goodness: float = 0.0
    heuristic_h: float = 0.0
    met_threshold: bool = False
    fail_count: int = 0
    failed_at: Optional[FailureStage] = None
    error_summary: str = ""
    verification_result: Any = None
    compilation_result: Any = None
    eval_result: Optional[EvaluationResult] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "parent_id": self.parent_id,
            "attempt_number": self.attempt_number,
            "goodness": self.goodness,
            "heuristic_h": self.heuristic_h,
            "met_threshold": self.met_threshold,
            "fail_count": self.fail_count,
            "failed_at": self.failed_at.value if self.failed_at else None,
            "strategy_analysis": None,
        }


@dataclass
class SearchTree:
    """Growing tree of candidate CSDs."""

    nodes: dict[int, SearchNode] = field(default_factory=dict)
    _next_id: int = 0

    def add_node(
        self,
        *,
        parent_id: Optional[int],
        attempt_number: int,
        strategy_code: str,
        full_dafny_code: str,
        timestamp: str,
        goodness: float = 0.0,
        met_threshold: bool = False,
        failed_at: Optional[FailureStage] = None,
        error_summary: str = "",
        verification_result: Any = None,
        compilation_result: Any = None,
        eval_result: Optional[EvaluationResult] = None,
    ) -> SearchNode:
        node_id = self._next_id
        self._next_id += 1
        h = heuristic_h(goodness)
        node = SearchNode(
            node_id=node_id,
            parent_id=parent_id,
            attempt_number=attempt_number,
            strategy_code=strategy_code,
            full_dafny_code=full_dafny_code,
            timestamp=timestamp,
            goodness=goodness,
            heuristic_h=h,
            met_threshold=met_threshold,
            failed_at=failed_at,
            error_summary=error_summary,
            verification_result=verification_result,
            compilation_result=compilation_result,
            eval_result=eval_result,
        )
        self.nodes[node_id] = node
        return node

    def all_nodes(self) -> list[SearchNode]:
        return list(self.nodes.values())

    def export(self) -> list[dict[str, Any]]:
        return [node.to_dict() for node in sorted(self.nodes.values(), key=lambda n: n.node_id)]

    def best_by_goodness(self) -> SearchNode:
        if not self.nodes:
            raise ValueError("SearchTree is empty")

        def sort_key(node: SearchNode) -> tuple[float, float, int]:
            accuracy = 0.0
            if node.eval_result is not None:
                accuracy = float(node.eval_result.accuracy or 0.0)
            return (node.goodness, accuracy, -node.node_id)

        return max(self.nodes.values(), key=sort_key)
