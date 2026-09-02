"""Regression tests for skipping pathological GSM expressions before expensive scoring."""

from types import SimpleNamespace

from synthesis.evaluate import evaluator as evaluator_module
from synthesis.evaluate.evaluator import Evaluator
from synthesis.evaluate.benchmarks.gsm_symbolic import eval_logic


def test_gsm_equivalence_rejects_pathological_expression_before_proof(monkeypatch):
    evaluator = Evaluator.__new__(Evaluator)
    pathological = " + ".join(["n_1"] * 90)
    proof_called = False

    def fail_if_called(*args, **kwargs):
        nonlocal proof_called
        proof_called = True
        raise AssertionError("pathological expression reached CRANE/Z3 proof")

    monkeypatch.setattr(
        evaluator_module,
        "_crane_validate_expression_equivalence",
        fail_if_called,
    )

    assert not evaluator._gsm_symbolic_equivalence(
        pathological,
        "n_1",
        {"n_1": "int"},
    )
    assert proof_called is False


def test_gsm_syntax_rejects_pathological_final_block_before_parser():
    pathological = "<<{}>>".format(" + ".join(["n_1"] * 90))
    evaluator = SimpleNamespace()

    def fail_if_called():
        raise AssertionError("pathological final block requested parser")

    evaluator._get_grammar_text = fail_if_called

    parses, segments = eval_logic.check_syntax(evaluator, pathological, {"variable_types": {"n_1": "int"}})

    assert parses is False
    assert segments == [(pathological, False)]
