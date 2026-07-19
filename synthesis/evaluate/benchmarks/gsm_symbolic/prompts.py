from __future__ import annotations

from typing import Any

from synthesis.evaluate.benchmarks.prompt_profiles import prompt_data_for_strategy


def _profile(strategy: str) -> dict[str, Any]:
    return prompt_data_for_strategy("gsm_symbolic", strategy)


GSM_CRANE_COT_TASK = str(_profile("crane")["task"])


def _fewshots(strategy: str) -> list[dict[str, str]]:
    profile = _profile(strategy)
    return list(profile["fewshots"])


def reasoning_with_symbolic_expr_prompt(question: str) -> str:
    """Flatten the YAML CRANE CoT profile into one prompt string."""
    parts = [GSM_CRANE_COT_TASK + "\n"]
    for example in _fewshots("crane"):
        parts.append(f"\n{example['question']}\n\n{example['answer']}\n")
    parts.append(f"\n{question}\n")
    return "".join(parts)


def reasoning_with_symbolic_expr_messages(question: str) -> list[dict[str, str]]:
    """Render the YAML CRANE CoT profile as system and few-shot chat turns."""
    messages = [{"role": "system", "content": GSM_CRANE_COT_TASK}]
    for example in _fewshots("crane"):
        messages.append({"role": "user", "content": example["question"]})
        messages.append({"role": "assistant", "content": example["answer"]})
    messages.append({"role": "user", "content": question})
    return messages


def symbolic_expression_only_prompt(question: str) -> str:
    """Flatten the YAML direct profile for GCD and IterGen."""
    profile = _profile("itergen")
    parts = [str(profile["task"]) + "\n"]
    for example in profile["fewshots"]:
        parts.append(f"\n{example['question']}\n\n{example['answer']}\n")
    parts.append(f"\n{question}\n")
    return "".join(parts)
