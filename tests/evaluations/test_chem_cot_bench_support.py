from pathlib import Path

import pytest
from lark import Lark

from evaluation.evaluator import Evaluator


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _build_parser() -> Lark:
    grammar = (PROJECT_ROOT / "utils" / "grammars" / "chem_cot_bench.lark").read_text(
        encoding="utf-8"
    )
    return Lark(grammar, start="start", parser="lalr")


def test_chem_grammar_accepts_smiles_like_answers():
    parser = _build_parser()

    tree = parser.parse("CC(=O)O")

    assert tree is not None


def test_chem_grammar_rejects_multiline_answers():
    parser = _build_parser()

    with pytest.raises(Exception):
        parser.parse("CCO\nwith explanation")


def test_chem_extraction_uses_last_constrained_span():
    evaluator = Evaluator(dataset_name="chem_cot_bench")

    actual = evaluator._extract_answer_chem_cot_bench("reasoning << CCO >> more text << CCN >>")

    assert actual == "CCN"


def test_chem_matching_normalizes_numbers_and_casefolds_text():
    evaluator = Evaluator(dataset_name="chem_cot_bench")

    assert evaluator._answers_match("42.0", "42")
    assert evaluator._answers_match("Copper Sulfate", "copper sulfate")


def test_chem_matching_supports_unordered_list_style_answers():
    evaluator = Evaluator(dataset_name="chem_cot_bench")
    example = {"matching_strategy": "unordered_set"}

    assert evaluator._answers_match("NaCl; H2O", "H2O; NaCl", example=example)


def test_chem_extra_tokens_optionally_include_full_prompt():
    evaluator = Evaluator(dataset_name="chem_cot_bench")
    dataset = [
        {
            "question": "Predict the product SMILES for ethanol oxidation.",
            "answer": "CC=O",
            "task": "reaction",
        }
    ]

    tokens = set(evaluator._collect_chem_cot_bench_extra_token_strings(dataset))

    assert "CC=O" in tokens
    assert any("You are solving a chemistry benchmark problem." in token for token in tokens)
