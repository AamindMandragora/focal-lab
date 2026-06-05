"""SynthesisPipeline defaults exercised by the matrix runner."""

from __future__ import annotations

import pytest

from tests.conftest import dummy_pipeline


class _UnloadCountingEvaluator:
    def __init__(self):
        self.unload_calls = 0

    def unload_runtime(self):
        self.unload_calls += 1


class _BackendGenerator:
    def __init__(self, backend):
        self.backend = backend


def test_pipeline_defaults_use_ucb_budget_and_refinement_beam():
    pipeline = dummy_pipeline()
    assert pipeline.helper_selection_policy == "bandit"
    assert pipeline.eval_max_seconds_per_example == 90.0
    assert pipeline.min_examples_before_threshold_stop == 15
    assert pipeline.refinement_beam_size == 2


def test_pipeline_keeps_evaluator_runtime_warm_for_hosted_author_models():
    from synthesis.evaluate.feedback_loop import SynthesisPipeline

    evaluator = _UnloadCountingEvaluator()
    pipeline = SynthesisPipeline(
        evaluator=evaluator,
        generator=_BackendGenerator("anthropic"),
        verifier=object(),
        compiler=object(),
    )
    pipeline._unload_evaluator_runtime_before_refinement()
    assert evaluator.unload_calls == 0


def test_pipeline_unloads_evaluator_runtime_for_local_author_models():
    from synthesis.evaluate.feedback_loop import SynthesisPipeline

    evaluator = _UnloadCountingEvaluator()
    pipeline = SynthesisPipeline(
        evaluator=evaluator,
        generator=_BackendGenerator("vllm"),
        verifier=object(),
        compiler=object(),
    )
    pipeline._unload_evaluator_runtime_before_refinement()
    assert evaluator.unload_calls == 1


def test_pipeline_rejects_utility_helper_policy():
    with pytest.raises(ValueError, match="bandit"):
        dummy_pipeline(helper_selection_policy="utility")
