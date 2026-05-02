import json
from pathlib import Path
from unittest.mock import Mock
import tempfile

import pytest


def test_evaluation_result_thresholds_and_summary():
    from evaluation.evaluator import EvaluationResult

    result = EvaluationResult(
        success=True,
        accuracy=0.8,
        format_rate=0.9,
        syntax_rate=0.95,
        num_examples=10,
        num_correct=8,
        total_time_seconds=5.0,
        sample_outputs=[
            {
                "question": "What is 5+3?",
                "expected": "8",
                "actual": "7",
                "is_correct": False,
            }
        ],
    )

    assert result.meets_threshold(0.7, 0.8, 0.9)
    assert not result.meets_threshold(0.9, 0.8, 0.9)
    summary = result.get_feedback_summary()
    assert "Accuracy: 80.0%" in summary
    assert "Sample Failures:" in summary
    as_dict = result.to_dict()
    assert as_dict["syntax_rate"] == 0.95


def test_evaluator_helpers_match_current_surface():
    from evaluation.evaluator import Evaluator

    evaluator = Evaluator(dataset_name="gsm_symbolic")
    assert evaluator._extract_constrained_content("hello <<5+3=8>> world") == ["5+3=8"]
    assert evaluator._extract_answer_gsm("hello <<5+3>> world") == "8"
    assert evaluator._extract_answer_gsm("hello <<5+3=8>> world") == "8"
    assert evaluator._extract_answer_gsm("hello <<16 * 8\n+ 4 * 10\n+ 13>> world") == "181"
    assert evaluator._check_format_validity("hello <<5+3=8>> world")
    assert not evaluator._check_format_validity("hello >>5+3=8<< world")
    assert evaluator._check_syntax_validity("hello <<16 * 8.5 + 4 * 10.5 + 13>> world")[0]
    assert not evaluator._check_syntax_validity("hello <<1>> world")[0]
    assert evaluator._get_grammar_file().name == "gsm.lark"


def test_evaluator_gsm_expression_evaluation_uses_delimited_segments_left_to_right():
    from evaluation.evaluator import Evaluator

    evaluator = Evaluator(dataset_name="gsm_symbolic")

    assert evaluator._extract_answer_gsm("reasoning <<1+1>> trailing <<16 * 8.5 + 4 * 10.5 + 13>>") == "191"
    assert (
        evaluator._extract_answer_gsm("reasoning <<x_1 = 3 + 2 * 6>> trailing <<y_1 = 5 + 10>> final <<x_1 / y_1>>")
        == "1"
    )
    assert evaluator._extract_answer_gsm("reasoning <<x_1 = 48 / 2>> final <<48 + x_1 + 0>>") == "72"
    assert evaluator._extract_answer_gsm("reasoning <<5+3>> trailing <<not valid>>") is None
    assert evaluator._answers_match("191", "191.0")


def test_gsm_parser_allows_decimal_prefix_tokens():
    from evaluation.common.parser_utils import create_lark_native_parser
    from generation.csd import VerifiedAgentSynthesis as VAS

    grammar = Path("utils/grammars/gsm.lark").read_text()
    parser_cls = create_lark_native_parser(grammar, VAS, start="csd_numeric_start")
    parser = parser_cls(["8", ".", "5", "+", "0"])

    assert "." in parser.ValidNextTokens(["8"])
    assert parser.IsValidPrefix(["8", "."])
    assert "5" in parser.ValidNextTokens(["8", "."])
    assert parser.IsCompletePrefix(["8", ".", "5", "+", "0", "+", "0"])


def test_gsm_parser_allows_scratch_assignment_prefix_tokens():
    from evaluation.common.parser_utils import create_lark_native_parser
    from generation.csd import VerifiedAgentSynthesis as VAS

    grammar = Path("utils/grammars/gsm.lark").read_text()
    parser_cls = create_lark_native_parser(grammar, VAS, start="csd_start")
    parser = parser_cls(["x", "_", "1", "=", "3", "+", "2", "*", "6"])

    assert "_" in parser.ValidNextTokens(["x"])
    assert parser.IsValidPrefix(["x", "_"])
    assert "1" in parser.ValidNextTokens(["x", "_"])
    assert parser.IsCompletePrefix(["x", "_", "1", "=", "3", "+", "2", "*", "6"])


def test_gsm_prompt_contains_worked_reasoning_example():
    from evaluation.evaluator import Evaluator

    evaluator = Evaluator(dataset_name="gsm_symbolic")
    prompt = evaluator._format_prompt({"question": "A box has 6 red pens and 4 blue pens. How many pens?"})

    assert "Worked GSM-style example" in prompt
    assert "Natalia sold clips to 48" in prompt
    assert "In May, she sold half as many as in April" in prompt
    assert "<<48 + 48 / 2>>" in prompt
    assert "<<x_1 = 48 / 2>>" in prompt
    assert "<<48 + x_1 + 0>>" in prompt
    assert "Reasoning checklist for the current problem" in prompt
    assert "changing rates over time" in prompt
    assert "repeated growth" in prompt
    assert "discounts" in prompt
    assert "total-cost questions" in prompt
    assert "budget questions about friends" in prompt
    assert "one-line finality check" in prompt
    assert "not just <intermediate quantity>" in prompt
    assert "Do not copy a worked-example expression" in prompt
    assert "may interleave plain-text reasoning" in prompt
    assert "Prefer a complete arithmetic expression" in prompt
    assert "Do not use a lone numeral" in prompt
    assert "<<8 + 0>>" in prompt
    assert "Copy numeric values exactly" in prompt


def test_gsm_csd_grammar_excludes_closing_delimiter_from_answer():
    from lark import Lark, UnexpectedInput

    grammar = Path("utils/grammars/gsm.lark").read_text()
    parser = Lark(grammar, start="csd_start", parser="lalr")
    numeric_parser = Lark(grammar, start="csd_numeric_start", parser="lalr")

    parser.parse("16 * 8.5 + 4 * 10.5 + 13")
    parser.parse("121 - 16 - 56")
    parser.parse("x_1 = 3 + 2 * 6")
    parser.parse("x_1 / y_1")
    with pytest.raises(UnexpectedInput):
        parser.parse("16 * 8.5 + 4 * 10.5 + 13 >>")
    with pytest.raises(UnexpectedInput):
        parser.parse("16 * 8")
    with pytest.raises(UnexpectedInput):
        parser.parse("8")
    with pytest.raises(UnexpectedInput):
        parser.parse("1")


def test_gsm_dynamic_grammar_keeps_scratch_vars_but_blocks_unbound_dataset_vars():
    from lark import Lark, UnexpectedInput
    from evaluation.gsm_symbolic.grammar import build_dynamic_grammar

    grammar = Path("utils/grammars/gsm.lark").read_text()
    dynamic = build_dynamic_grammar(grammar, [])
    parser = Lark(dynamic, start="csd_start", parser="lalr")

    parser.parse("x_1 = 3 + 2 * 6")
    parser.parse("x_1 + 4 + 0")
    with pytest.raises(UnexpectedInput):
        parser.parse("x + 13 - 5")


def test_failure_stage_and_attempt_record_evaluation_results():
    from evaluation.evaluator import EvaluationResult
    from synthesis.feedback_loop import FailureStage, SynthesisAttempt
    from synthesis.runner import RuntimeResult
    from verification.verifier import VerificationResult

    attempt = SynthesisAttempt(
        attempt_number=1,
        strategy_code="strategy",
        timestamp="2026-04-14T00:00:00",
        verification_result=VerificationResult(success=True, raw_output="ok"),
        runtime_result=RuntimeResult(success=True, output=["x"], cost=1),
        eval_result=EvaluationResult(
            success=True,
            accuracy=0.4,
            format_rate=0.5,
            syntax_rate=0.6,
            num_examples=5,
            num_correct=2,
            total_time_seconds=1.0,
        ),
        failed_at=FailureStage.EVALUATION,
        error_summary="evaluation failed",
        generation_diagnostics=[{"candidate": 1, "raw_output_empty": False}],
    )

    attempt_dict = attempt.to_dict()
    assert FailureStage.EVALUATION.value == "evaluation"
    assert attempt_dict["failed_at"] == "evaluation"
    assert attempt_dict["evaluation"]["accuracy"] == 0.4
    assert attempt_dict["generation_diagnostics"][0]["candidate"] == 1


def test_successful_attempt_does_not_require_dafny_compilation():
    from evaluation.evaluator import EvaluationResult
    from synthesis.feedback_loop import SynthesisAttempt
    from synthesis.runner import RuntimeResult
    from verification.verifier import VerificationResult

    attempt = SynthesisAttempt(
        attempt_number=1,
        strategy_code="strategy",
        timestamp="2026-04-14T00:00:00",
        verification_result=VerificationResult(success=True, raw_output="ok"),
        runtime_result=RuntimeResult(success=True, output=["x"], cost=1),
        eval_result=EvaluationResult(
            success=True,
            accuracy=1.0,
            format_rate=1.0,
            syntax_rate=1.0,
            num_examples=1,
            num_correct=1,
            total_time_seconds=1.0,
        ),
    )

    assert attempt.succeeded()


def test_pipeline_requires_evaluator_and_stores_thresholds():
    from synthesis.feedback_loop import SynthesisPipeline

    with pytest.raises(TypeError):
        SynthesisPipeline()

    pipeline = SynthesisPipeline(
        evaluator=Mock(),
        generator=Mock(),
        verifier=Mock(),
        compiler=Mock(),
        runner=Mock(),
        min_accuracy=0.7,
        min_format_rate=0.8,
        min_syntax_rate=0.9,
        eval_sample_size=4,
        save_reports=False,
    )

    assert pipeline.min_accuracy == 0.7
    assert pipeline.min_format_rate == 0.8
    assert pipeline.min_syntax_rate == 0.9
    assert pipeline.eval_sample_size == 4


def test_build_evaluation_failure_prompt_and_generator_refinement():
    from generation.generator import StrategyGenerator
    from generation.prompts import build_evaluation_failure_prompt

    system_prompt, user_prompt = build_evaluation_failure_prompt(
        previous_strategy="old strategy",
        evaluation_feedback="Accuracy: 30%",
    )
    assert system_prompt
    assert "old strategy" in user_prompt
    assert "Accuracy: 30%" in user_prompt

    generator = StrategyGenerator.__new__(StrategyGenerator)
    generator._generate_valid_strategy = Mock(return_value="new strategy")
    refined = generator.refine_after_evaluation_failure("old strategy", "Accuracy: 30%")
    generator._generate_valid_strategy.assert_called_once()
    assert refined == "new strategy"


def test_generator_raises_instead_of_using_canned_fallback():
    from generation.generator import StrategyGenerationError, StrategyGenerator

    generator = StrategyGenerator.__new__(StrategyGenerator)
    generator.max_new_tokens = 192
    generator.temperature = 0.7
    generator._generate_text = Mock(return_value="not a strategy")
    generator._extract_strategy = Mock(return_value="not a strategy")
    generator._ensure_rationale_block = Mock(side_effect=ValueError("missing rationale"))

    with pytest.raises(StrategyGenerationError, match="usable initial strategy"):
        generator._generate_valid_strategy(
            "system",
            "user",
            failure_context="Qwen did not produce a usable initial strategy",
        )

    assert not hasattr(StrategyGenerator, "STARTER_STRATEGY")


def test_pipeline_retries_after_evaluation_failure():
    from evaluation.evaluator import EvaluationResult
    from synthesis.feedback_loop import FailureStage, SynthesisPipeline
    from synthesis.runner import RuntimeResult
    from verification.verifier import VerificationResult

    mock_generator = Mock()
    mock_generator.generate_initial = Mock(return_value="strategy_initial")
    mock_generator.refine_after_evaluation_failure = Mock(return_value="strategy_refined")
    mock_generator.inject_strategy = Mock(return_value="full python code")

    mock_verifier = Mock()
    mock_verifier.verify = Mock(return_value=VerificationResult(success=True, raw_output="ok"))

    mock_runner = Mock()
    mock_runner.run_python_native = Mock(return_value=RuntimeResult(success=True, output=["token"], cost=1))

    mock_evaluator = Mock()
    eval_results = [
        EvaluationResult(
            success=True,
            accuracy=0.2,
            format_rate=0.3,
            syntax_rate=0.4,
            num_examples=3,
            num_correct=1,
            total_time_seconds=1.0,
        ),
        EvaluationResult(
            success=True,
            accuracy=0.8,
            format_rate=0.9,
            syntax_rate=0.95,
            num_examples=3,
            num_correct=3,
            total_time_seconds=1.0,
        ),
    ]
    mock_evaluator.evaluate_sample = Mock(side_effect=eval_results)

    with tempfile.TemporaryDirectory() as tmpdir:
        pipeline = SynthesisPipeline(
            evaluator=mock_evaluator,
            generator=mock_generator,
            verifier=mock_verifier,
            compiler=None,
            runner=mock_runner,
            max_iterations=3,
            output_dir=Path(tmpdir),
            save_reports=False,
            min_accuracy=0.5,
            min_format_rate=0.5,
            min_syntax_rate=0.5,
        )

        result = pipeline.synthesize("test task", output_name="test_csd")

    assert result.success
    assert result.python_source_path is not None
    assert result.python_source_path.name == "GeneratedCSD.py"
    assert len(result.attempts) == 2
    assert result.attempts[0].failed_at == FailureStage.EVALUATION
    assert result.attempts[1].failed_at is None
    assert mock_generator.refine_after_evaluation_failure.call_count == 1
    assert mock_evaluator.evaluate_sample.call_count == 2


def test_success_report_includes_evaluation_metrics():
    from evaluation.evaluator import EvaluationResult
    from synthesis.feedback_loop import SynthesisPipeline
    from synthesis.runner import RuntimeResult
    from verification.verifier import VerificationResult

    mock_generator = Mock()
    mock_generator.generate_initial = Mock(return_value="strategy_initial")
    mock_generator.inject_strategy = Mock(return_value="full python code")

    mock_verifier = Mock()
    mock_verifier.verify = Mock(return_value=VerificationResult(success=True, raw_output="ok"))

    mock_runner = Mock()
    mock_runner.run_python_native = Mock(return_value=RuntimeResult(success=True, output=["token"], cost=1))

    mock_eval_result = EvaluationResult(
        success=True,
        accuracy=0.7,
        format_rate=0.9,
        syntax_rate=1.0,
        num_examples=10,
        num_correct=7,
        total_time_seconds=12.0,
    )
    mock_evaluator = Mock()
    mock_evaluator.evaluate_sample = Mock(return_value=mock_eval_result)

    with tempfile.TemporaryDirectory() as tmpdir:
        pipeline = SynthesisPipeline(
            evaluator=mock_evaluator,
            generator=mock_generator,
            verifier=mock_verifier,
            compiler=None,
            runner=mock_runner,
            max_iterations=1,
            output_dir=Path(tmpdir),
            save_reports=True,
            min_accuracy=0.6,
            min_format_rate=0.8,
            min_syntax_rate=0.9,
        )

        result = pipeline.synthesize("test task", output_name="test_csd")
        assert result.success

        report_path = result.run_dir / "success_report.json"
        assert report_path.exists()
        report = json.loads(report_path.read_text())
        assert report["evaluation"]["accuracy"] == 0.7
        assert report["evaluation"]["format_rate"] == 0.9
        assert report["evaluation"]["syntax_rate"] == 1.0
        assert report["evaluation"]["num_examples"] == 10
        assert report["evaluation"]["num_correct"] == 7
