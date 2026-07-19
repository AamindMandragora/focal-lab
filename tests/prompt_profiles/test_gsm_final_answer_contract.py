from __future__ import annotations

from pathlib import Path

from synthesis.evaluate.benchmarks.gsm_symbolic import eval_logic


class _Evaluator:
    def _get_grammar_text(self) -> str:
        grammar_path = Path(__file__).parents[2] / "synthesis" / "evaluate" / "grammars" / "gsm.lark"
        return grammar_path.read_text()

    @staticmethod
    def _gsm_symbolic_equivalence(actual: str | None, expected: str, variable_types: dict) -> bool:
        return actual == expected


def test_gsm_evaluator_accepts_the_final_answer_sentence() -> None:
    evaluator = _Evaluator()
    example = {
        "answer_parsed": "x + 2",
        "variable_types": {"x": "int"},
    }
    output = "Let's think step by step. <reasoning> The final answer is <<x + 2>>."

    actual, source, aux = eval_logic.extract_actual(evaluator, output, example)
    syntax_valid, segments = eval_logic.check_syntax(evaluator, output, example)
    correct = eval_logic.is_correct(evaluator, actual, "x + 2", example, aux, output)

    assert actual == "x + 2"
    assert source == "last_visible_span"
    assert syntax_valid is True
    assert segments == [("<<x + 2>>", True)]
    assert correct is True

