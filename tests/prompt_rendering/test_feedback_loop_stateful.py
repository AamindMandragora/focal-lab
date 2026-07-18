"""Golden (characterization) tests for the two stateful SynthesisPipeline builders.

  - SynthesisPipeline._get_verification_history_summary  (uses no self)
  - SynthesisPipeline._build_attempt_outcome_ledger      (uses self.generator + extract_rationale)

We avoid SynthesisPipeline.__init__ (which needs a generator + evaluator + GPU) by
constructing a bare instance with object.__new__ and setting only the attributes the
method under test touches. `_get_verification_history_summary` reads no self at all, so
it is called with a throwaway instance too.

REFACTOR GUARDS: GREEN on current code, must STAY byte-identical after conversion.
Regenerate: REGEN_GOLDENS=1 pytest <thisfile>  (only against known-good current code).
"""
import os
import pathlib

import pytest

from synthesis.evaluate.feedback_loop import (
    FailureStage,
    SynthesisAttempt,
    SynthesisPipeline,
)
from synthesis.evaluate.evaluator import EvaluationResult
from synthesis.verify.verifier import VerificationResult, VerificationDiagnostic

GOLDEN_DIR = pathlib.Path(__file__).parent / "fixtures" / "feedback_loop_stateful"


def _check(name: str, produced: str):
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    path = GOLDEN_DIR / f"{name}.golden.txt"
    if os.environ.get("REGEN_GOLDENS"):
        path.write_text(produced)
        pytest.skip(f"regenerated {path.name}")
    assert path.read_text() == produced


def _bare_pipeline(**attrs):
    """A SynthesisPipeline instance with __init__ skipped; only `attrs` are set."""
    p = object.__new__(SynthesisPipeline)
    for k, v in attrs.items():
        setattr(p, k, v)
    return p


def _diag(**kw):
    base = dict(file="Gen.dfy", line=0, column=0, message="msg", obligation_kind="postcondition")
    base.update(kw)
    return VerificationDiagnostic(**base)


def _eval(accuracy, syntax_rate, num_examples, num_correct, sample_outputs):
    return EvaluationResult(
        success=True, accuracy=accuracy, contains_delimiters=True, syntax_rate=syntax_rate,
        num_examples=num_examples, num_correct=num_correct, total_time_seconds=1.0,
        sample_outputs=sample_outputs,
    )


def _attempt(n, *, strategy_code="// s", failed_at=None, error_summary="",
             verification_result=None, eval_result=None):
    return SynthesisAttempt(
        attempt_number=n, strategy_code=strategy_code, full_dafny_code="// d",
        timestamp="2026-01-01T00:00:00", failed_at=failed_at, error_summary=error_summary,
        verification_result=verification_result, eval_result=eval_result,
    )


# ---------------------------------------------------------------------------
# _get_verification_history_summary
# ---------------------------------------------------------------------------

def test_verification_history_empty():
    attempts = [_attempt(1, failed_at=FailureStage.EVALUATION)]  # no VERIFICATION failures
    p = _bare_pipeline()
    _check("vhist__empty", p._get_verification_history_summary(attempts))


def test_verification_history_mixed():
    vr_diag = VerificationResult(
        success=False,
        diagnostics=[_diag(obligation_kind="postcondition", line=12, call_name="StepForward",
                           failing_text="cost := cost + 1",
                           related_file="/repo/lib/Helpers.dfy", related_line=7)],
    )
    vr_nodiag = VerificationResult(success=False, diagnostics=[])
    attempts = [
        _attempt(2, failed_at=FailureStage.VERIFICATION, verification_result=vr_diag),
        _attempt(3, failed_at=FailureStage.VERIFICATION, verification_result=vr_nodiag,
                 error_summary="First error line\nsecond line ignored"),
    ]
    p = _bare_pipeline()
    _check("vhist__mixed", p._get_verification_history_summary(attempts))


# ---------------------------------------------------------------------------
# _build_attempt_outcome_ledger
# ---------------------------------------------------------------------------

def test_outcome_ledger_empty():
    attempts = [_attempt(1)]  # no eval_result -> not evaluated
    p = _bare_pipeline(generator=object())
    _check("ledger__empty", p._build_attempt_outcome_ledger(attempts, None))


def test_outcome_ledger_full():
    best_strategy = "// CSD_RATIONALE_BEGIN\n// Force << early and close >>.\n// CSD_RATIONALE_END\nmethod B() {}"
    best_eval = _eval(
        0.5, 0.8, 4, 2,
        [
            {"is_correct": True},
            {"is_correct": False, "failure_location": "span_open"},
            {"is_correct": False, "failure_location": "span_open"},
            {"is_correct": False, "failure_location": "answer"},
        ],
    )
    recent_eval = _eval(
        0.25, 0.6, 4, 1,
        [{"is_correct": False, "failure_location": "unknown"}],
    )
    attempts = [
        _attempt(1, strategy_code="method R() {}", eval_result=recent_eval),  # no rationale markers
        _attempt(2, strategy_code=best_strategy, eval_result=best_eval),
    ]
    p = _bare_pipeline(generator=object())  # no summarize_rationale_claim -> raw rationale
    _check("ledger__full", p._build_attempt_outcome_ledger(attempts, best_attempt_number=2))
