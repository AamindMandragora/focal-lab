"""YAML-backed prompt profiles for fixed CSD baseline adapters."""

from __future__ import annotations

from functools import lru_cache
import logging
from pathlib import Path
from typing import Any

import yaml


LOGGER = logging.getLogger(__name__)
_PROFILE_DIR = Path(__file__).resolve().parent
_DATASET_FILES = {
    "spider": "sql.yaml",
    "sql": "sql.yaml",
    "smiles": "smiles.yaml",
    "gsm": "gsm/profile.yaml",
    "gsm_symbolic": "gsm/profile.yaml",
}
_FIXED_CSD_STRATEGIES = {"gcd", "itergen", "crane"}


@lru_cache(maxsize=None)
def _load_prompt_config(dataset: str) -> dict[str, Any]:
    normalized = dataset.strip().lower()
    filename = _DATASET_FILES.get(normalized)
    if filename is None:
        raise ValueError(f"No fixed-CSD prompt YAML for dataset: {dataset}")

    path = _PROFILE_DIR / filename
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError(f"Prompt YAML must contain a mapping: {path}")

    strategy_profiles = config.get("strategy_profiles")
    profiles = config.get("profiles")
    if not isinstance(strategy_profiles, dict) or not isinstance(profiles, dict):
        raise ValueError(f"Prompt YAML needs strategy_profiles and profiles mappings: {path}")
    if set(strategy_profiles) != _FIXED_CSD_STRATEGIES:
        raise ValueError(
            f"Prompt YAML must map exactly {sorted(_FIXED_CSD_STRATEGIES)}: {path}"
        )
    if strategy_profiles["gcd"] != strategy_profiles["itergen"]:
        raise ValueError(f"GCD and IterGen must share one direct prompt profile: {path}")
    if strategy_profiles["crane"] != "chain_of_thought":
        raise ValueError(f"CRANE must use the chain_of_thought prompt profile: {path}")
    for strategy, profile_name in strategy_profiles.items():
        if profile_name not in profiles:
            raise ValueError(
                f"Strategy {strategy} references missing profile {profile_name!r}: {path}"
            )

    LOGGER.debug(
        "[fixed-csd-prompt] loaded dataset=%s source=%s mappings=%s",
        normalized,
        path,
        strategy_profiles,
    )
    return config


def prompt_profile_for_strategy(dataset: str, strategy: str) -> str:
    """Return the YAML-selected profile name for one fixed CSD strategy."""
    normalized_strategy = strategy.strip().lower()
    if normalized_strategy not in _FIXED_CSD_STRATEGIES:
        raise ValueError(f"Unknown fixed CSD strategy: {strategy}")
    return str(_load_prompt_config(dataset)["strategy_profiles"][normalized_strategy])


def prompt_data_for_strategy(dataset: str, strategy: str) -> dict[str, Any]:
    """Return one validated profile, including materialized GSM few-shots."""
    config = _load_prompt_config(dataset)
    profile_name = prompt_profile_for_strategy(dataset, strategy)
    raw_profile = config["profiles"][profile_name]
    if not isinstance(raw_profile, dict):
        raise ValueError(f"Prompt profile {profile_name!r} must be a mapping")
    profile = dict(raw_profile)

    task_template = profile.get("task_template")
    if task_template is not None:
        if not isinstance(task_template, str):
            raise ValueError(f"Prompt profile {profile_name!r} task_template must be text")
        values = profile.get("values", {})
        if not isinstance(values, dict):
            raise ValueError(f"Prompt profile {profile_name!r} values must be a mapping")
        profile["task"] = task_template.format_map(values)

    examples = config.get("examples")
    if examples is not None:
        if not isinstance(profile.get("task"), str):
            raise ValueError(f"Prompt profile {profile_name!r} needs a task string")
        if not isinstance(examples, list) or not examples:
            raise ValueError("GSM prompt YAML needs a non-empty examples list")
        answer_field = profile.get("answer_field")
        question_field = profile.get("question_field")
        if not isinstance(answer_field, str):
            raise ValueError(f"Prompt profile {profile_name!r} needs answer_field")
        fewshots = []
        for index, example in enumerate(examples, start=1):
            if not isinstance(example, dict):
                raise ValueError(f"GSM example {index} must be a mapping")
            question = example.get(question_field) if isinstance(question_field, str) else None
            question = question or example.get("question")
            answer = example.get(answer_field)
            if not isinstance(question, str) or not isinstance(answer, str):
                raise ValueError(
                    f"GSM example {index} needs text question and {answer_field!r} answer"
                )
            fewshots.append({"question": question, "answer": answer})
        profile["fewshots"] = fewshots

    return profile


def render_strategy_prompt(
    dataset: str,
    strategy: str,
    example: dict[str, Any],
) -> str:
    """Render the exact prompt selected by the dataset YAML and CSD strategy."""
    profile_name = prompt_profile_for_strategy(dataset, strategy)
    profile = prompt_data_for_strategy(dataset, strategy)
    if not isinstance(profile, dict) or not isinstance(profile.get("template"), str):
        raise ValueError(f"Prompt profile {profile_name!r} needs a string template")

    normalized_dataset = dataset.strip().lower()
    if normalized_dataset in {"spider", "sql"}:
        fields = {
            "db_id": example.get("db_id", ""),
            "db_info": example.get("db_info", ""),
            "question": example.get("question", ""),
        }
    else:
        fields = {"base_prompt": str(example.get("prompt", "")).rstrip()}

    values = profile.get("values", {})
    if not isinstance(values, dict):
        raise ValueError(f"Prompt profile {profile_name!r} values must be a mapping")
    fields.update(values)
    rendered = profile["template"].format_map(fields)
    if profile.get("append_trailing_space") is True:
        rendered += " "

    LOGGER.debug(
        "[fixed-csd-prompt] rendered dataset=%s strategy=%s profile=%s chars=%d",
        normalized_dataset,
        strategy,
        profile_name,
        len(rendered),
    )
    return rendered
