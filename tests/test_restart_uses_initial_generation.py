"""Restart-mode prompt behavior.

When the synthesis loop decides to restart, it should request a fresh initial
generation again. There should be no separate restart prompt surface.
"""

from __future__ import annotations

import inspect

from synthesis.evaluate.feedback_loop import SynthesisPipeline as EvalSynthesisPipeline
from synthesis.generate.generator import StrategyGenerator
from synthesis.generate import prompts


def test_restart_prompt_entrypoints_are_removed():
    assert not hasattr(prompts, "EVALUATION_FAILURE_RESTART_PROMPT")
    assert not hasattr(prompts, "build_evaluation_failure_restart_prompt")
    assert not hasattr(StrategyGenerator, "refine_after_evaluation_failure_restart")


def test_restart_branches_use_initial_generation_prompt():
    eval_source = inspect.getsource(EvalSynthesisPipeline)

    assert "refine_after_evaluation_failure_restart" not in eval_source
    assert ".generate_initial(" in eval_source
