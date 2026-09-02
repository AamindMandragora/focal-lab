"""Golden (characterization) tests for the two remaining EvaluationResult prose builders:

  - get_behavioral_context_summary  ("\n".join(lines), empty guard "")
  - _render_mode_examples           ("\n\n".join(blocks), empty guards "")

(get_feedback_summary is builder #8, already converted and covered elsewhere. The
per-line List[str] `_summarize_*` helpers stay Python — a template already lays their
lines out inside #8 — consistent with the verify/ and feedback_loop precedent.)

REFACTOR GUARDS: GREEN on current code, must STAY byte-identical after conversion.
Regenerate: REGEN_GOLDENS=1 pytest <thisfile>  (only against known-good current code).
"""
import os
import pathlib

import pytest

from synthesis.evaluate.evaluator import EvaluationResult

GOLDEN_DIR = pathlib.Path(__file__).parent / "fixtures" / "evaluator_prose"


def _check(name: str, produced: str):
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    path = GOLDEN_DIR / f"{name}.golden.txt"
    if os.environ.get("REGEN_GOLDENS"):
        path.write_text(produced)
        pytest.skip(f"regenerated {path.name}")
    assert path.read_text() == produced


def _result(sample_outputs):
    return EvaluationResult(
        success=True, accuracy=0.5, contains_delimiters=True, syntax_rate=0.75,
        num_examples=len(sample_outputs), num_correct=0, total_time_seconds=1.0,
        sample_outputs=sample_outputs,
    )


# ---------------------------------------------------------------------------
# get_behavioral_context_summary
# ---------------------------------------------------------------------------

_TRACED_SAMPLE = {
    "helper_trace": [
        {"helper": "ForceOpen", "detail": "forced <<", "cost_before": 0, "cost_after": 1},
        {"helper": "CloseSpan", "detail": "closed >>"},
    ],
    "provenance_tags": ["x"],  # present -> skips the mutating _annotate call
    "token_count": 42,
    "contains_delimiters": True,
    "syntax_rate": 0.75,
    "answer_provenance": "span",
    "failure_location": "answer",
}


def test_behavioral_context_empty():
    r = _result([{"actual": "x", "is_correct": False}])  # no helper_trace
    _check("behavctx__empty", r.get_behavioral_context_summary())


def test_behavioral_context_with_delimiters():
    r = _result([dict(_TRACED_SAMPLE)])
    _check("behavctx__with_delimiters", r.get_behavioral_context_summary(require_delimiters=True))


def test_behavioral_context_no_delimiters():
    r = _result([dict(_TRACED_SAMPLE)])
    _check("behavctx__no_delimiters", r.get_behavioral_context_summary(require_delimiters=False))


# ---------------------------------------------------------------------------
# _render_mode_examples
# ---------------------------------------------------------------------------

def test_render_mode_examples_empty():
    r = _result([])
    _check("render_mode__empty", r._render_mode_examples())


def test_render_mode_examples_full():
    # Two `error` samples -> both classify as runtime_or_generation_error (single mode);
    # distinct full_output lengths -> deterministic shortest->longest pick order (2 blocks).
    s1 = {"error": "RuntimeError: boom", "full_output": "short",
          "question_full": "Full prompt one", "question": "Q1", "actual": None, "expected": "42"}
    s2 = {"error": "ValueError: nope", "full_output": "a much longer qwen output here",
          "question": "Q2", "actual": "7", "expected": "7"}  # no question_full -> falls back to question
    _check("render_mode__full", _result([s1, s2])._render_mode_examples())
