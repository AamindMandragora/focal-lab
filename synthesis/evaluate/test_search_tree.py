"""Unit tests for explicit search tree."""

from synthesis.evaluate.evaluator import EvaluationResult
from synthesis.evaluate.search_tree import SearchTree


def test_parent_child_linkage_and_export():
    tree = SearchTree()
    root = tree.add_node(
        parent_id=None,
        attempt_number=1,
        strategy_code="root",
        full_dafny_code="full_root",
        timestamp="t1",
        goodness=0.2,
    )
    child = tree.add_node(
        parent_id=root.node_id,
        attempt_number=2,
        strategy_code="child",
        full_dafny_code="full_child",
        timestamp="t2",
        goodness=0.5,
    )
    exported = tree.export()
    assert len(exported) == 2
    assert exported[0]["node_id"] == root.node_id
    assert exported[1]["parent_id"] == root.node_id
    assert child.parent_id == root.node_id


def test_best_by_goodness_tiebreaks_on_accuracy():
    tree = SearchTree()
    eval_low = EvaluationResult(
        success=True,
        accuracy=0.3,
        contains_delimiters=True,
        syntax_rate=0.9,
        num_examples=10,
        num_correct=3,
        total_time_seconds=1.0,
    )
    eval_high = EvaluationResult(
        success=True,
        accuracy=0.5,
        contains_delimiters=True,
        syntax_rate=0.9,
        num_examples=10,
        num_correct=5,
        total_time_seconds=1.0,
    )
    tree.add_node(
        parent_id=None,
        attempt_number=1,
        strategy_code="a",
        full_dafny_code="fa",
        timestamp="t1",
        goodness=0.8,
        eval_result=eval_low,
    )
    best = tree.add_node(
        parent_id=0,
        attempt_number=2,
        strategy_code="b",
        full_dafny_code="fb",
        timestamp="t2",
        goodness=0.8,
        eval_result=eval_high,
    )
    assert tree.best_by_goodness().node_id == best.node_id
