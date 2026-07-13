"""Tests for no-gold candidate consensus selection.

The selector is meant to codify the H27/H28 lesson: agreement across candidate
answers should beat an isolated clean-looking candidate, while quality score is
still the tie-breaker inside an agreement cluster. These tests intentionally use
only candidate text, source metadata, no-gold equivalence keys, and no-gold
quality scores. Expected answers never appear in the selector inputs.
"""

from synthesis.evaluate.candidate_consensus import Candidate, select_consensus


def test_agreement_beats_isolated_higher_score_candidate():
    result = select_consensus([
        Candidate(
            group_id="gsm-1",
            expression="wrong_but_clean",
            equivalence_key="wrong_but_clean",
            source="attempt-1",
            source_family="single",
            quality_score=0.99,
        ),
        Candidate(
            group_id="gsm-1",
            expression="supported_expr_a",
            equivalence_key="supported_expr",
            source="attempt-2",
            source_family="variant",
            quality_score=0.70,
        ),
        Candidate(
            group_id="gsm-1",
            expression="supported_expr_b",
            equivalence_key="supported_expr",
            source="attempt-3",
            source_family="variant",
            quality_score=0.72,
        ),
    ])

    chosen = result["gsm-1"]
    assert chosen.candidate.expression == "supported_expr_b"
    assert chosen.cluster_key == "supported_expr"
    assert chosen.candidate_count == 2
    assert chosen.source_count == 2


def test_source_family_diversity_breaks_equal_candidate_count_tie():
    result = select_consensus([
        Candidate("gsm-2", "same_family_1", "same", "h24-a", "h24", 0.91),
        Candidate("gsm-2", "same_family_2", "same", "h24-b", "h24", 0.92),
        Candidate("gsm-2", "cross_family_1", "cross", "h2", "h2", 0.80),
        Candidate("gsm-2", "cross_family_2", "cross", "h10", "h10", 0.81),
    ])

    chosen = result["gsm-2"]
    assert chosen.cluster_key == "cross"
    assert chosen.candidate.expression == "cross_family_2"
    assert chosen.family_count == 2


def test_quality_score_breaks_tie_inside_chosen_cluster():
    result = select_consensus([
        Candidate("gsm-3", "low_quality", "agreed", "a", "fam-a", 0.10),
        Candidate("gsm-3", "high_quality", "agreed", "b", "fam-b", 0.95),
    ])

    chosen = result["gsm-3"]
    assert chosen.candidate.expression == "high_quality"
    assert chosen.best_quality_score == 0.95


def test_empty_candidate_list_returns_empty_selection():
    assert select_consensus([]) == {}


if __name__ == "__main__":
    test_agreement_beats_isolated_higher_score_candidate()
