"""Behavioral tests for prompt-only grounding and SMILES checks."""

import sys
import types
from unittest.mock import MagicMock

import rdkit  # noqa: F401


for _heavy in ("torch", "transformers", "vllm"):
    if _heavy not in sys.modules:
        try:
            __import__(_heavy)
        except Exception:
            sys.modules[_heavy] = MagicMock()

from synthesis.evaluate.benchmarks.common.model_utils import (  # noqa: E402
    _TensorizedLMBase,
    _candidate_smiles,
    _candidate_identifiers,
    _parse_schema_support,
    _smiles_resemblance,
)


_PROMPT = (
    "db_info: # singer ( singer_id , name , country , age )\n"
    "question: How many singers?\n"
    "db_info: # employee ( emp_id , emp_name , dept )\n"
    "# department ( dept_id , dept_name )\n"
    "question: List employee names.\n"
)
_PROMPT_NO_SCHEMA = "Solve the math word problem.\nquestion: What is 2 plus 2?\n"


class _StubLM(_TensorizedLMBase):
    def __init__(self, instruction_text):
        self.instruction_text = instruction_text

    def _to_str(self, value):
        return value if isinstance(value, str) else "".join(value)


def test_parse_schema_support_isolates_real_schema():
    support = _parse_schema_support(_PROMPT)
    for name in ("employee", "emp_id", "emp_name", "dept", "department", "dept_id", "dept_name"):
        assert name in support
    for leaked in ("singer", "singer_id", "country", "age", "name"):
        assert leaked not in support


def test_parse_schema_support_empty_when_no_schema():
    assert _parse_schema_support(_PROMPT_NO_SCHEMA) == set()
    assert _parse_schema_support("") == set()


def test_candidate_identifiers_strips_literals_keywords_aliases():
    candidates = _candidate_identifiers("SELECT emp_name FROM employee WHERE dept = 'Sales'")
    assert {"emp_name", "employee", "dept"} <= set(candidates)
    assert not {"sales", "select", "from", "where"} & set(candidates)
    aliased = _candidate_identifiers("SELECT t1.emp_name FROM employee AS t1")
    assert "t1" not in aliased


def test_span_grounded_true_for_in_schema():
    lm = _StubLM(_PROMPT)
    assert lm.SpanGrounded("SELECT emp_name FROM employee") is True
    assert lm.SpanGrounded("SELECT dept_name FROM department") is True


def test_span_grounded_false_for_out_of_schema():
    lm = _StubLM(_PROMPT)
    assert lm.SpanGrounded("SELECT bogus_col FROM employee") is False
    assert lm.SpanGrounded("SELECT emp_name FROM nonexistent_table") is False


def test_span_grounded_allows_aliases_and_literals():
    lm = _StubLM(_PROMPT)
    query = "SELECT t1.emp_name FROM employee AS t1 WHERE t1.dept = 'Engineering'"
    assert lm.SpanGrounded(query) is True


def test_span_grounded_noop_without_schema():
    assert _StubLM(_PROMPT_NO_SCHEMA).SpanGrounded("anything goes") is True


def test_span_grounded_cache_is_stable():
    lm = _StubLM(_PROMPT)
    first = lm._grounding_support_set()
    second = lm._grounding_support_set()
    assert first == second and first is second


def test_span_appears_in_prompt_detects_labeled_smiles_example():
    lm = _StubLM("Generate valid molecules.\nMolecule: CCO\nMolecule: CCN\n")
    assert lm.SpanAppearsInPrompt(" <<CCO>> ") is True
    assert lm.SpanAppearsInPrompt("CCN") is True


def test_span_appears_in_prompt_detects_rolling_suffix_smiles():
    lm = _StubLM("Generate valid molecules.\nO=C=Nc1ccccc1\nMolecule:\n")
    assert lm.SpanAppearsInPrompt("O=C=Nc1ccccc1") is True


def test_span_appears_in_prompt_rejects_substring_false_positive():
    lm = _StubLM("Generate valid molecules.\nMolecule: CCO\n")
    assert lm.SpanAppearsInPrompt("CC") is False
    assert lm.SpanAppearsInPrompt("CCO") is True


def test_span_appears_in_prompt_rejects_empty_or_label_only_text():
    lm = _StubLM("Generate valid molecules.\nMolecule: CCO\n")
    assert lm.SpanAppearsInPrompt("") is False
    assert lm.SpanAppearsInPrompt("Molecule:") is False


_ACRYLATE_A = "C=CC(=O)OCC"
_ACRYLATE_B = "C=CC(=O)OCCCC"
_ISOCYANATE = "O=C=Nc1ccccc1"


def test_candidate_smiles_strips_label_and_delimiters():
    assert _candidate_smiles("Molecule: CCO") == "CCO"
    assert _candidate_smiles("<<CCN>>") == "CCN"
    assert _candidate_smiles("  `C=CC(=O)OCC`  ") == _ACRYLATE_A
    assert _candidate_smiles("") == ""
    assert _candidate_smiles("CCO extra tokens") == "CCO"


def test_resemblance_identical_candidate_scores_one():
    candidate, score = _smiles_resemblance(_ACRYLATE_A, [_ACRYLATE_A, _ISOCYANATE])
    assert candidate == _ACRYLATE_A
    assert abs(score - 1.0) < 1e-9


def test_resemblance_similar_beats_dissimilar():
    _, close = _smiles_resemblance(_ACRYLATE_B, [_ACRYLATE_A])
    _, far = _smiles_resemblance(_ISOCYANATE, [_ACRYLATE_A])
    assert close > far


def test_resemblance_zero_without_exemplars():
    candidate, score = _smiles_resemblance(_ACRYLATE_A, [])
    assert candidate == _ACRYLATE_A
    assert score == 0.0


def test_resemblance_zero_for_unparseable_candidate():
    _, score = _smiles_resemblance("not_a_molecule)))(", [_ACRYLATE_A])
    assert score == 0.0


def test_resemblance_empty_candidate():
    assert _smiles_resemblance("", [_ACRYLATE_A]) == ("", 0.0)


def test_span_resemblance_class_method_uses_prompt_examples():
    lm = _StubLM(f"Generate a molecule.\nMolecule: {_ACRYLATE_A}\n")
    lm._dafny = types.SimpleNamespace(BigRational=float)
    assert lm.SpanResemblanceToPromptExamples(_ACRYLATE_B) > lm.SpanResemblanceToPromptExamples(_ISOCYANATE)


def test_span_resemblance_zero_without_prompt_examples():
    lm = _StubLM("Solve the math word problem.\n")
    lm._dafny = types.SimpleNamespace(BigRational=float)
    assert lm.SpanResemblanceToPromptExamples(_ACRYLATE_A) == 0.0
