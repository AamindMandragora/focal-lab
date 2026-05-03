from pathlib import Path

import pytest
from lark import Lark

from synthesis.evaluator import Evaluator


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _build_parser() -> Lark:
    grammar = (PROJECT_ROOT / "grammars" / "smiles.lark").read_text(encoding="utf-8")
    return Lark(grammar, start="start", parser="lalr")


def test_smiles_grammar_accepts_smiles_like_answers():
    parser = _build_parser()
    tree = parser.parse("CC(=O)O")
    assert tree is not None


def test_smiles_grammar_rejects_multiline_answers():
    parser = _build_parser()
    with pytest.raises(Exception):
        parser.parse("CCO\nwith explanation")


def test_evaluator_supports_smiles():
    evaluator = Evaluator(dataset_name="smiles")
    assert evaluator.dataset_name == "smiles"


def test_extract_answer_smiles_uses_last_constrained_span():
    evaluator = Evaluator(dataset_name="smiles")
    actual = evaluator._extract_answer_smiles("reasoning << CCO >> more text << CCN >>")
    assert actual == "CCN"


def test_smiles_matching_normalizes_numbers_and_casefolds_text():
    evaluator = Evaluator(dataset_name="smiles")
    assert evaluator._answers_match("42.0", "42")
    assert evaluator._answers_match("Copper Sulfate", "copper sulfate")


def test_smiles_matching_supports_unordered_list_style_answers():
    evaluator = Evaluator(dataset_name="smiles")
    example = {"matching_strategy": "unordered_set"}
    assert evaluator._answers_match("NaCl; H2O", "H2O; NaCl", example=example)
