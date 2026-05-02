from pathlib import Path
import json
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from evaluations.gsm_symbolic.dataset import enrich_gsm_symbolic_example
from evaluations.gsm_symbolic.grammar import extract_variables_from_mapping
from synthesis.evaluator import Evaluator


def test_extract_variables_filters_out_string_slots():
    mapping = {"n": "str", "x": "int", "rate": "float", "big_fish": "str", "k": "int"}
    assert extract_variables_from_mapping(mapping) == ["x", "rate", "k"]
    assert extract_variables_from_mapping(mapping, include_string_variables=True) == ["n", "x", "rate", "big_fish", "k"]


def test_enrich_gsm_symbolic_example_backfills_crane_metadata(tmp_path: Path):
    payload = {
        "question": "Benny saw a 10-foot shark with 2 6-inch remoras attached to it. What percentage of the shark's body length is the combined length of the remoras?",
        "question_parsed": "{n} saw a {x}-foot {big_fish} with {k} {y}-inch remoras attached to it. What percentage of the {big_fish}'s body length is the combined length of the remoras?",
        "answer_parsed": "int((k * y)/ (x * 12) * 100)",
        "variable_types": {"n": "str", "x": "int", "big_fish": "str", "k": "int", "y": "int"},
        "id_orig": 473,
        "id_shuffled": 0,
    }
    (tmp_path / '0000.json').write_text(json.dumps(payload))

    example = {
        "question": "Rania saw a 210-foot whale with 7 72-inch remoras attached to it. What percentage of the whale's body length is the combined length of the remoras?",
        "original_question": payload["question"],
        "original_id": 473,
    }

    enriched = enrich_gsm_symbolic_example(example, crane_dir=tmp_path)
    assert enriched["variable_types"] == payload["variable_types"]
    assert enriched["question_parsed"] == payload["question_parsed"]
    assert enriched["answer_parsed"] == payload["answer_parsed"]


def test_gsm_prompt_surfaces_allowed_variables_and_symbolic_template():
    evaluator = Evaluator(dataset_name="gsm_symbolic")
    example = {
        "question": "Rania saw a 210-foot whale with 7 72-inch remoras attached to it.",
        "question_parsed": "{n} saw a {x}-foot {big_fish} with {k} {y}-inch remoras attached to it.",
        "variable_types": {"n": "str", "x": "int", "big_fish": "str", "k": "int", "y": "int"},
    }
    prompt = evaluator._format_prompt(example)
    assert "x, k, y" in prompt
    assert "Do not invent new variable names outside this set." in prompt
    assert example["question_parsed"] in prompt


def test_gsm_syntax_validation_uses_example_specific_variable_set():
    evaluator = Evaluator(dataset_name="gsm_symbolic")
    example = {"variable_types": {"n": "str", "x": "int", "k": "int", "y": "int"}}

    is_valid, segments = evaluator._check_syntax_validity("<<x + k>>", example=example)
    assert is_valid is True
    assert segments == [("x + k", True)]

    is_valid, segments = evaluator._check_syntax_validity("<<o_new + k>>", example=example)
    assert is_valid is False
    assert segments == [("o_new + k", False)]
