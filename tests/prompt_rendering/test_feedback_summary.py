"""Golden (characterization) test for builder #8:
EvaluationResult.get_feedback_summary.

These goldens were captured from the LIVE method on real success_report.json
samples (see fixtures/*.json). This is a REFACTOR guard, not new behavior:
the test is GREEN on the current code and must STAY GREEN after the builder is
converted to the pydantic-model + Jinja-template pattern. Any byte difference
is a regression (or an intended, separately-reviewed content revision).
"""
import json
import pathlib

import pytest

from synthesis.evaluate.evaluator import EvaluationResult

FIX = pathlib.Path(__file__).parent / "fixtures"

CASES = [
    ("feedback_case_gsm.json", True, "feedback_case_gsm_reqdelim_true.golden.txt"),
    ("feedback_case_gsm.json", False, "feedback_case_gsm_reqdelim_false.golden.txt"),
    ("feedback_case_smiles.json", False, "feedback_case_smiles_reqdelim_false.golden.txt"),
]


@pytest.mark.parametrize("fixture,require_delimiters,golden", CASES)
def test_get_feedback_summary_matches_golden(fixture, require_delimiters, golden):
    kwargs = json.loads((FIX / fixture).read_text())
    result = EvaluationResult(**kwargs)
    rendered = result.get_feedback_summary(require_delimiters=require_delimiters)
    expected = (FIX / golden).read_text()
    assert rendered == expected
