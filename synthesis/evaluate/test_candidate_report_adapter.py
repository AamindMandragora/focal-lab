"""Tests for converting direct-eval reports into no-gold consensus candidates."""

from synthesis.evaluate.candidate_consensus import Candidate
from synthesis.evaluate.candidate_report_adapter import (
    candidates_from_direct_eval_report,
    candidates_from_direct_eval_reports,
)


def test_direct_eval_report_builds_no_gold_candidates_and_skips_unusable_outputs():
    report = {
        "source_id": "h40_t0",
        "output_name": "h37_gsm2b_repeat_probe_20260629_t0",
        "sample_outputs": [
            {
                "actual": " n + 1 ",
                "has_extracted_answer": True,
                "is_syntax_valid": True,
                "failure_location": "syntax_valid_semantic_mismatch",
                "answer_source": "last_visible_span",
                "visible_span_token_lengths": [3],
                "expected": "gold must not be used",
                "is_correct": False,
            },
            {
                "actual": "missing answer",
                "has_extracted_answer": False,
                "is_syntax_valid": True,
                "expected": "gold must not be used",
                "is_correct": True,
            },
            {
                "actual": "invalid syntax",
                "has_extracted_answer": True,
                "is_syntax_valid": False,
                "expected": "gold must not be used",
                "is_correct": True,
            },
        ],
    }

    candidates = candidates_from_direct_eval_report(report, source_family="temp_sweep")

    assert candidates == [
        Candidate(
            group_id=1,
            expression="n + 1",
            equivalence_key="n + 1",
            source="h40_t0",
            source_family="temp_sweep",
            quality_score=0.8,
        )
    ]


def test_multiple_reports_keep_distinct_source_metadata():
    reports = [
        {
            "source_id": "h40_t0",
            "output_name": "h37_gsm2b_repeat_probe_20260629_t0",
            "sample_outputs": [
                {
                    "actual": "a + b",
                    "has_extracted_answer": True,
                    "is_syntax_valid": True,
                    "failure_location": "correct",
                }
            ],
        },
        {
            "source_id": "h42_t2",
            "output_name": "h37_gsm2b_repeat_probe_20260629_t2",
            "sample_outputs": [
                {
                    "actual": "a + b",
                    "has_extracted_answer": True,
                    "is_syntax_valid": True,
                    "failure_location": "syntax_valid_semantic_mismatch",
                }
            ],
        },
    ]

    candidates = candidates_from_direct_eval_reports(reports)

    assert [candidate.source for candidate in candidates] == ["h40_t0", "h42_t2"]
    assert [candidate.source_family for candidate in candidates] == [
        "h37_gsm2b_repeat_probe_20260629_t0",
        "h37_gsm2b_repeat_probe_20260629_t2",
    ]
    assert [candidate.equivalence_key for candidate in candidates] == ["a + b", "a + b"]


def test_direct_eval_report_can_add_candidate_lines_from_visible_text():
    report = {
        "source_id": "h24",
        "output_name": "h24_gsm2b_multicandidate_selector_probe_20260629",
        "sample_outputs": [
            {
                "actual": "chosen answer",
                "has_extracted_answer": True,
                "is_syntax_valid": True,
                "failure_location": "syntax_valid_semantic_mismatch",
                "full_output": "\n".join([
                    "Let's compare candidates.",
                    "Candidate A: {g} - {n_1} - 3{n_2}",
                    "Candidate B: $g - n_1 - n_2 - 2n_2$ (same expression)",
                    "Candidate A is the most concise answer.",
                    "1. This numbered reasoning step is not a candidate.",
                ]),
                "scored_output": "Candidate A: {g} - {n_1} - 3{n_2}",
                "expected": "gold must not be used",
                "is_correct": False,
            }
        ],
    }

    candidates = candidates_from_direct_eval_report(
        report,
        source_family="candidate_line_pool",
        include_candidate_lines=True,
    )

    assert candidates == [
        Candidate(
            group_id=1,
            expression="chosen answer",
            equivalence_key="chosen answer",
            source="h24",
            source_family="candidate_line_pool",
            quality_score=0.8,
        ),
        Candidate(
            group_id=1,
            expression="{g} - {n_1} - 3{n_2}",
            equivalence_key="{g} - {n_1} - 3{n_2}",
            source="h24:candidate_line:A",
            source_family="candidate_line_pool:candidate_lines",
            quality_score=0.75,
        ),
        Candidate(
            group_id=1,
            expression="$g - n_1 - n_2 - 2n_2$",
            equivalence_key="$g - n_1 - n_2 - 2n_2$",
            source="h24:candidate_line:B",
            source_family="candidate_line_pool:candidate_lines",
            quality_score=0.75,
        ),
    ]
