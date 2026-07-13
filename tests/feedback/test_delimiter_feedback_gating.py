from synthesis.evaluate.evaluator import EvaluationResult


def _sample(output: str) -> dict:
    return {
        "question": "q",
        "question_full": "q",
        "expected": "SELECT 1",
        "actual": "SELECT 2",
        "full_output": output,
        "scored_output": output,
        "is_correct": False,
        "accuracy_applicable": True,
        "contains_delimiters": True,
        "visible_delimiters": True,
        "used_constrained_chunk": False,
        "uses_hidden_chunks": False,
        "is_syntax_valid": True,
        "syntax_rate": 1.0,
        "runtime_budget_exceeded": False,
        "num_visible_spans": 1,
        "num_valid_visible_spans": 1,
        "has_extracted_answer": True,
        "token_count": 10,
        "time_seconds": 1.0,
        "hit_max_steps": False,
    }


def _result() -> EvaluationResult:
    return EvaluationResult(
        success=False,
        accuracy=0.0,
        contains_delimiters=True,
        syntax_rate=1.0,
        num_examples=1,
        num_correct=0,
        total_time_seconds=1.0,
        sample_outputs=[_sample("SELECT <<x>> FROM t")],
    )


DELIMITER_MARKERS = [
    "Contains << >>",
    "Structural Generation Metrics",
    "Examples with visible `<<`",
    "Answer extraction source",
    "Span usefulness",
]
GENERAL_MARKERS = ["Accuracy:", "Syntax Rate:", "Correctness by syntax bucket"]


def test_delimiter_diagnostics_present_when_required():
    summary = _result().get_feedback_summary(require_delimiters=True)
    assert all(marker in summary for marker in DELIMITER_MARKERS + GENERAL_MARKERS)


def test_default_keeps_delimiter_diagnostics():
    summary = _result().get_feedback_summary()
    assert "Structural Generation Metrics" in summary
    assert "Contains << >>" in summary


def test_delimiter_diagnostics_omitted_when_not_required():
    summary = _result().get_feedback_summary(require_delimiters=False)
    assert all(marker not in summary for marker in DELIMITER_MARKERS)
    assert all(marker in summary for marker in GENERAL_MARKERS)
