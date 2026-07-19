from __future__ import annotations

from pathlib import Path

from scripts.runtime import run_cold_synthesis_queue as queue
from synthesis.evaluate.benchmarks.gsm_symbolic import eval_logic
from synthesis.evaluate.benchmarks.gsm_symbolic import prompts
from synthesis.evaluate.benchmarks.prompt_profiles import (
    prompt_data_for_strategy,
    prompt_profile_for_strategy,
)
from synthesis.evaluate.run_legacy_fixed_strategy import _fixed_csd_prompt


QUESTION = "If there are {c} cars and {nc} more arrive, how many are there?"


def test_gsm_yaml_is_the_single_profile_source_for_all_three_csds() -> None:
    profile_path = (
        Path(prompts.__file__).parents[1]
        / "prompt_profiles"
        / "gsm"
        / "profile.yaml"
    )

    assert profile_path.is_file()
    assert prompt_profile_for_strategy("gsm_symbolic", "gcd") == "direct"
    assert prompt_profile_for_strategy("gsm_symbolic", "itergen") == "direct"
    assert prompt_profile_for_strategy("gsm_symbolic", "crane") == "chain_of_thought"


def test_gsm_yaml_contains_complete_direct_and_cot_profiles() -> None:
    direct = prompt_data_for_strategy("gsm_symbolic", "itergen")
    crane = prompt_data_for_strategy("gsm_symbolic", "crane")

    assert len(direct["fewshots"]) == 8
    assert len(crane["fewshots"]) == 8
    assert direct["fewshots"][0]["answer"] == "<<tf - t>>"
    assert crane["fewshots"][-1]["answer"].endswith(
        "The final answer is <<m - q * p>>."
    )
    assert "Only output the symbolic expression wrapped in << >>" in direct["task"]
    assert "The final answer is <<symbolic expression>>" in crane["task"]


def test_gsm_runtime_and_queue_render_from_yaml_profiles() -> None:
    direct = prompt_data_for_strategy("gsm_symbolic", "gcd")
    crane = prompt_data_for_strategy("gsm_symbolic", "crane")
    example = {"question_parsed": QUESTION}

    direct_prompt = eval_logic.format_prompt_for_strategy(None, example, "gcd")
    crane_messages = eval_logic.format_prompt_for_strategy(None, example, "crane")

    assert isinstance(direct_prompt, str)
    assert direct_prompt.startswith(direct["task"] + "\n")
    assert direct_prompt.endswith("\n" + QUESTION + "\n")
    assert isinstance(crane_messages, list)
    assert crane_messages[0] == {"role": "system", "content": crane["task"]}
    assert crane_messages[-1] == {"role": "user", "content": QUESTION}
    assert len(crane_messages) == 18
    assert queue.GSM_TASK == crane["task"] == prompts.GSM_CRANE_COT_TASK


def test_fixed_csd_dispatch_uses_gsm_yaml_strategy_mapping() -> None:
    example = {"question_parsed": QUESTION}

    gcd = _fixed_csd_prompt(eval_logic, None, example, "gcd")
    itergen = _fixed_csd_prompt(eval_logic, None, example, "itergen")
    crane = _fixed_csd_prompt(eval_logic, None, example, "crane")

    assert gcd == itergen
    assert isinstance(crane, list)
    assert crane[0]["content"] == prompt_data_for_strategy(
        "gsm_symbolic", "crane"
    )["task"]

