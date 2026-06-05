"""Tests for metadecode prompt-tier selection from compiled CSD bodies."""

from __future__ import annotations

from pathlib import Path

import pytest

from synthesis.evaluate.prompt_tiers import (
    configure_eval_prompts,
    prompt_tier_for_csd_strategy,
    strategy_uses_reasoning_prompt,
)

_REF = Path(__file__).resolve().parents[1] / "synthesis" / "verify" / "reference"


@pytest.fixture
def gcd_strategy() -> str:
    return (_REF / "gcd.dfy").read_text()


@pytest.fixture
def crane_strategy() -> str:
    return (_REF / "crane.dfy").read_text()


def test_fully_constrained_reference_uses_tier_one(gcd_strategy: str) -> None:
    assert strategy_uses_reasoning_prompt(gcd_strategy) is False
    assert prompt_tier_for_csd_strategy(gcd_strategy) == 1


def test_reasoning_reference_uses_tier_two(crane_strategy: str) -> None:
    assert strategy_uses_reasoning_prompt(crane_strategy) is True
    assert prompt_tier_for_csd_strategy(crane_strategy) == 2


def test_crane_generation_helper_uses_tier_two() -> None:
    body = 'generated := helpers.CraneGeneration(lm, parser, prompt, maxSteps, 10, eosToken);'
    assert strategy_uses_reasoning_prompt(body) is True


def test_constrained_generation_helper_uses_tier_one() -> None:
    body = 'generated, _ := helpers.ConstrainedGeneration(lm, parser, prompt, maxSteps, eosToken);'
    assert strategy_uses_reasoning_prompt(body) is False


class _FakeEvaluator:
    def __init__(self, dataset_name: str) -> None:
        self.dataset_name = dataset_name
        self.prompt_tier = 2
        self.use_reasoning_prompt = None
        self.grammar_prompt_tier = None
        self.smiles_delimited_answer_prompt = False


def test_configure_eval_prompts_sets_gsm_tier_from_strategy(gcd_strategy: str, crane_strategy: str) -> None:
    ev = _FakeEvaluator("gsm_symbolic")
    configure_eval_prompts(ev, gcd_strategy)
    assert ev.prompt_tier == 1
    assert ev.use_reasoning_prompt is False
    assert ev.grammar_prompt_tier is None

    configure_eval_prompts(ev, crane_strategy)
    assert ev.prompt_tier == 2
    assert ev.use_reasoning_prompt is True


def test_configure_eval_prompts_smiles_keeps_delimited_grammar_for_tier_one(
    gcd_strategy: str,
) -> None:
    ev = _FakeEvaluator("smiles")
    configure_eval_prompts(ev, gcd_strategy)
    assert ev.prompt_tier == 1
    assert ev.grammar_prompt_tier == 2
    assert ev.smiles_delimited_answer_prompt is True
