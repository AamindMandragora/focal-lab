from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_gsm_scoring_has_no_project_specific_size_rejection():
    syntax_source = (
        REPO_ROOT / "synthesis/evaluate/benchmarks/gsm_symbolic/eval_logic.py"
    ).read_text()
    correctness_source = (
        REPO_ROOT / "synthesis/evaluate/evaluator.py"
    ).read_text()

    marker = "_is_pathological_gsm_scoring_expression"
    assert marker not in syntax_source
    assert marker not in correctness_source
