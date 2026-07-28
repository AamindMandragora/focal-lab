"""Golden (characterization) tests for the verify/ feedback builders.

Covers the three model-facing text builders in synthesis/verify/*, per-branch:
  - CompilationResult.get_error_summary   (compiler.py)
  - VerificationResult.get_error_summary  (verifier.py)
  - VerificationResult.get_structured_feedback (verifier.py) — exercises
    _remediation_for / _cited_contract_line through the rendered remediation line.

These are REFACTOR GUARDS: GREEN on the current code, and must STAY byte-identical
after each builder is converted to the pydantic-model + Jinja-template pattern.
Any byte difference is a regression (these carry NO descriptive change).

Regenerate goldens from the current implementation with:  REGEN_GOLDENS=1 pytest <thisfile>
(only run REGEN against known-good current code, never after a conversion).
"""
import os
import pathlib

import pytest

from synthesis.verify.compiler import CompilationError, CompilationResult
from synthesis.verify.verifier import (
    VerificationDiagnostic,
    VerificationError,
    VerificationResult,
)

GOLDEN_DIR = pathlib.Path(__file__).parent / "fixtures" / "verify"


def _check(name: str, produced: str):
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    path = GOLDEN_DIR / f"{name}.golden.txt"
    if os.environ.get("REGEN_GOLDENS"):
        path.write_text(produced)
        pytest.skip(f"regenerated {path.name}")
    assert path.read_text() == produced


# ---------------------------------------------------------------------------
# CompilationResult.get_error_summary
# ---------------------------------------------------------------------------

COMPILE_CASES = {
    "compile_success": CompilationResult(success=True, output_dir="/outputs/generated/run1"),
    "compile_no_errors_raw_output": CompilationResult(
        success=False, raw_output="raw compiler blob\nsecond line", raw_stderr="stderr blob"
    ),
    "compile_no_errors_stderr_only": CompilationResult(
        success=False, raw_output="", raw_stderr="only stderr present"
    ),
    "compile_with_errors": CompilationResult(
        success=False,
        errors=[
            CompilationError("F.dfy", 10, 3, "undefined identifier x"),
            CompilationError("F.dfy", 22, 1, "type mismatch: expected int, got string"),
        ],
    ),
}


@pytest.mark.parametrize("name", sorted(COMPILE_CASES))
def test_compilation_get_error_summary_matches_golden(name):
    _check(f"compile__{name}", COMPILE_CASES[name].get_error_summary())


# ---------------------------------------------------------------------------
# VerificationResult.get_error_summary
# ---------------------------------------------------------------------------

VERIFY_SUMMARY_CASES = {
    "verify_success": VerificationResult(success=True),
    "verify_raw_output": VerificationResult(
        success=False, raw_output="   Dafny(5,2): Error: postcondition\n  Related location   "
    ),
    "verify_no_raw_no_errors_stderr": VerificationResult(success=False, raw_stderr="stderr text here"),
    "verify_no_raw_no_errors_fallback": VerificationResult(success=False),
    "verify_with_errors": VerificationResult(
        success=False,
        errors=[
            VerificationError("F.dfy", 5, 2, "postcondition might not hold"),
            VerificationError("F.dfy", 9, 4, "cost might exceed maxSteps"),
        ],
    ),
}


@pytest.mark.parametrize("name", sorted(VERIFY_SUMMARY_CASES))
def test_verification_get_error_summary_matches_golden(name):
    _check(f"verify_summary__{name}", VERIFY_SUMMARY_CASES[name].get_error_summary())


# ---------------------------------------------------------------------------
# VerificationResult.get_structured_feedback  (+ _remediation_for branches)
# ---------------------------------------------------------------------------

def _diag(**kw):
    base = dict(file="F.dfy", line=12, column=3, message="msg", obligation_kind="postcondition")
    base.update(kw)
    return VerificationDiagnostic(**base)


STRUCTURED_CASES = {
    # short-circuits to "" ---------------------------------------------------
    "sf_empty_success": VerificationResult(
        success=True, diagnostics=[_diag()]
    ),
    "sf_empty_no_diagnostics": VerificationResult(success=False, diagnostics=[]),
    # postcondition remediation sub-branches (via marked '>' line) -----------
    "sf_post_progress_neq": VerificationResult(
        success=False,
        diagnostics=[_diag(contract_excerpt="  ensures\n> || generated != generatedPrefix ||\n  more")],
    ),
    "sf_post_cost_le": VerificationResult(
        success=False, diagnostics=[_diag(contract_excerpt="> cost <= maxSteps")]
    ),
    "sf_post_length": VerificationResult(
        success=False,
        diagnostics=[_diag(contract_excerpt="> |generated| <= |generatedPrefix| + maxSteps")],
    ),
    "sf_post_generic": VerificationResult(
        success=False, diagnostics=[_diag(contract_excerpt="> ensures SomeOtherClause(x)")]
    ),
    "sf_precondition": VerificationResult(
        success=False,
        diagnostics=[_diag(obligation_kind="precondition", call_name="AppendToken")],
    ),
    "sf_invariant": VerificationResult(
        success=False, diagnostics=[_diag(obligation_kind="invariant")]
    ),
    "sf_decreases": VerificationResult(
        success=False, diagnostics=[_diag(obligation_kind="decreases")]
    ),
    "sf_verification_fallback": VerificationResult(
        success=False, diagnostics=[_diag(obligation_kind="verification")]
    ),
    # every optional field present ------------------------------------------
    "sf_all_optional_fields": VerificationResult(
        success=False,
        diagnostics=[
            _diag(
                obligation_kind="postcondition",
                message="postcondition might not hold",
                call_name="ConstrainedStep",
                failing_text="return;",
                related_file="/proofs/VerifiedAgentSynthesis.dfy",
                related_line=88,
                related_message="This is the postcondition that might not hold",
                source_excerpt="line a\nline b",
                contract_excerpt="> cost <= maxSteps\n  decreases maxSteps",
            )
        ],
    ),
    # minimal diagnostic: only required fields ------------------------------
    "sf_minimal_diag": VerificationResult(
        success=False, diagnostics=[_diag(obligation_kind="invariant", contract_excerpt="")]
    ),
    # two diagnostics (index numbering) -------------------------------------
    "sf_two_diagnostics": VerificationResult(
        success=False,
        diagnostics=[
            _diag(obligation_kind="precondition", call_name="Foo"),
            _diag(obligation_kind="decreases", line=20),
        ],
    ),
}


@pytest.mark.parametrize("name", sorted(STRUCTURED_CASES))
def test_verification_get_structured_feedback_matches_golden(name):
    _check(f"structured__{name}", STRUCTURED_CASES[name].get_structured_feedback())
