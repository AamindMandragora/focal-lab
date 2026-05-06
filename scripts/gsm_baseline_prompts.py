from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from typing import Any


def _replace_markers(text: str) -> str:
    return text.replace("[[START]]", "<<").replace("[[END]]", ">>")


def load_crane_gsm_prompt_config(crane_repo: Path) -> dict[str, Any]:
    import yaml

    template_path = crane_repo.expanduser().resolve() / "src" / "prompt_templates" / "gsm_symbolic.yaml"
    if not template_path.exists():
        raise FileNotFoundError(f"CRANE GSM prompt template not found: {template_path}")
    config = yaml.safe_load(template_path.read_text())
    if not isinstance(config, dict):
        raise ValueError(f"CRANE GSM prompt template did not load as a mapping: {template_path}")
    return config


def crane_gsm_system_prompt(crane_repo: Path, *, instruct_type: str = "gsm") -> str:
    config = load_crane_gsm_prompt_config(crane_repo)
    task_spec = str(config.get("task_specification") or "").strip()
    cot_instruct = str(config.get("cot_instruct", {}).get(instruct_type) or "").strip()
    return _replace_markers(dedent(f"{task_spec}\n{cot_instruct}").strip())


def crane_gsm_fewshots(
    crane_repo: Path,
    *,
    instruct_type: str = "gsm",
    num_shots: int = 8,
) -> list[dict[str, str]]:
    config = load_crane_gsm_prompt_config(crane_repo)
    raw_fewshots = config.get("fewshots", {}).get("cot", {}).get(instruct_type, [])
    fewshots: list[dict[str, str]] = []
    for example in raw_fewshots[:num_shots]:
        fewshots.append({
            "question": str(example["question"]),
            "response": _replace_markers(str(example["response"])),
        })
    return fewshots


def crane_gsm_chat_prompt(
    crane_repo: Path,
    question: str,
    *,
    instruct_type: str = "gsm",
    num_shots: int = 8,
) -> list[dict[str, str]]:
    system_content = crane_gsm_system_prompt(crane_repo, instruct_type=instruct_type)
    messages: list[dict[str, str]] = []
    if system_content:
        messages.append({"role": "system", "content": system_content})
    for example in crane_gsm_fewshots(crane_repo, instruct_type=instruct_type, num_shots=num_shots):
        messages.append({"role": "user", "content": example["question"]})
        messages.append({"role": "assistant", "content": example["response"]})
    messages.append({"role": "user", "content": question})
    messages.append({"role": "assistant", "content": ""})
    return messages


def crane_gsm_text_prompt(
    crane_repo: Path,
    question: str,
    *,
    instruct_type: str = "gsm",
    num_shots: int = 8,
) -> str:
    parts: list[str] = []
    system_content = crane_gsm_system_prompt(crane_repo, instruct_type=instruct_type)
    if system_content:
        parts.append(system_content)
    for example in crane_gsm_fewshots(crane_repo, instruct_type=instruct_type, num_shots=num_shots):
        parts.append(f"Question:\n{example['question']}\n\nResponse:\n{example['response']}")
    parts.append(f"Question:\n{question}\n\nResponse:")
    return "\n\n".join(parts)
