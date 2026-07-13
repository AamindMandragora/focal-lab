"""Tests for CPU-only scoring of recovered GSM candidate lines."""

import json

from synthesis.evaluate.candidate_consensus import Candidate
from synthesis.evaluate.candidate_line_scorer import (
    load_gsm_split_examples,
    score_candidate_lines,
    score_gsm_expression,
)


def test_load_gsm_split_examples_uses_manifest_indices(tmp_path):
    crane_dir = tmp_path / "crane"
    crane_dir.mkdir()
    for idx, answer in enumerate(["a", "b", "c"]):
        (crane_dir / f"{idx:03d}.json").write_text(json.dumps({
            "question_parsed": f"question {idx}",
            "answer_parsed": answer,
            "variable_types": {"x": "int"},
        }))
    split_file = tmp_path / "split.json"
    split_file.write_text(json.dumps({
        "crane_dir": str(crane_dir),
        "train_indices": [2, 0],
    }))

    examples = load_gsm_split_examples(split_file, split_name="train")

    assert [example["question"] for example in examples] == ["question 2", "question 0"]
    assert [example["answer_parsed"] for example in examples] == ["c", "a"]
    assert [example["variable_types"] for example in examples] == [{"x": "int"}, {"x": "int"}]


def test_load_gsm_split_examples_falls_back_from_stale_manifest_path(
    tmp_path,
    monkeypatch,
):
    crane_dir = tmp_path / "portable-crane" / "src" / "gsm_symbolic"
    crane_dir.mkdir(parents=True)
    (crane_dir / "000.json").write_text(json.dumps({
        "question_parsed": "portable question",
        "answer_parsed": "x",
        "variable_types": {"x": "int"},
    }))
    split_file = tmp_path / "split.json"
    split_file.write_text(json.dumps({
        "crane_dir": "/home/someone-else/missing/CRANE/src/gsm_symbolic",
        "train_indices": [0],
    }))
    monkeypatch.setenv("CRANE_GSM_SYMBOLIC_DIR", str(crane_dir))

    examples = load_gsm_split_examples(split_file, split_name="train")

    assert [example["question"] for example in examples] == ["portable question"]


def test_score_gsm_expression_normalizes_placeholder_multiplication():
    example = {
        "answer_parsed": "g - n_1 - 3*n_2",
        "variable_types": {"g": "int", "n_1": "int", "n_2": "int"},
    }

    assert score_gsm_expression("{g} - {n_1} - 3{n_2}", example)
    assert not score_gsm_expression("{g} - {n_1} - 2{n_2}", example)


def test_score_candidate_lines_reports_union_and_selected_correctness():
    reports = [{
        "source_id": "toy_report",
        "output_name": "toy_report",
        "sample_outputs": [{
            "actual": "wrong",
            "has_extracted_answer": True,
            "is_syntax_valid": True,
            "failure_location": "syntax_valid_semantic_mismatch",
            "full_output": "\n".join([
                "Candidate 1: {g} - {n_1} - 2{n_2}",
                "Candidate 2: {g} - {n_1} - 3{n_2}",
            ]),
        }],
    }]
    examples = [{
        "answer_parsed": "g - n_1 - 3*n_2",
        "variable_types": {"g": "int", "n_1": "int", "n_2": "int"},
    }]

    result = score_candidate_lines(reports, examples)

    assert result.candidate_line_count == 2
    assert result.correct_union_indices == [1]
    assert result.selected_correct_indices == []
    assert [row.is_correct for row in result.rows] == [False, True]
    assert all(isinstance(row.candidate, Candidate) for row in result.rows)
