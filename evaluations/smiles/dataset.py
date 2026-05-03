"""
SMILES benchmark dataset loading utilities.

Loads chemistry answer-generation tasks from Chem-CoT-Bench and normalizes
them to a compact prompt/answer surface used by the evaluator.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence


DEFAULT_SMILES_CONFIGS = (
    "mol_und",
    "mol_edit",
    "mol_opt",
    "reaction",
)


@dataclass
class SmilesExample:
    """A single benchmark example normalized for evaluator use."""

    id: str
    question: str
    answer: str
    config: str
    task: str = ""
    matching_strategy: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def _datasets_offline_enabled() -> bool:
    return any(
        os.environ.get(name, "").strip() in {"1", "true", "True"}
        for name in ("HF_DATASETS_OFFLINE", "HF_HUB_OFFLINE")
    )


def _is_hf_connection_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "failed to resolve",
            "name or service not known",
            "temporary failure in name resolution",
            "connection error",
            "maxretryerror",
            "httpsconnectionpool",
            "offline mode",
        )
    )


def _first_present(item: dict[str, Any], keys: Iterable[str], default: Any = "") -> Any:
    for key in keys:
        value = item.get(key)
        if value not in (None, "", [], {}):
            return value
    return default


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        if value and all(isinstance(item, str) for item in value):
            return " | ".join(item.strip() for item in value if str(item).strip())
        return json.dumps(value, sort_keys=True, ensure_ascii=True)
    if isinstance(value, dict):
        if "content" in value and isinstance(value["content"], str):
            return value["content"].strip()
        return json.dumps(value, sort_keys=True, ensure_ascii=True)
    return str(value).strip()


def _messages_to_text(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    parts: list[str] = []
    for item in messages:
        if isinstance(item, dict):
            role = str(item.get("role", "")).strip()
            content = _stringify(item.get("content", ""))
            if content:
                parts.append(f"{role}: {content}" if role else content)
        else:
            text = _stringify(item)
            if text:
                parts.append(text)
    return "\n".join(parts).strip()


def _normalize_item(item: dict[str, Any], idx: int, config: str) -> SmilesExample | None:
    question = _stringify(
        _first_present(item, ("query", "question", "instruction", "prompt", "input"), "")
    )
    if not question and item.get("messages") is not None:
        question = _messages_to_text(item.get("messages"))

    answer = _stringify(
        _first_present(
            item,
            (
                "ground_truth",
                "answer",
                "target",
                "output",
                "label",
                "expected",
                "reference",
                "gold",
            ),
            "",
        )
    )

    if not question or not answer:
        return None

    task = _stringify(
        _first_present(
            item,
            ("task", "task_name", "subtask", "subset", "category", "file_name"),
            config,
        )
    )
    matching_strategy = _stringify(
        _first_present(item, ("matching_strategy", "metric", "answer_type"), "")
    )
    example_id = _stringify(
        _first_present(item, ("id", "sample_id", "uuid"), f"{config}-{idx}")
    )

    metadata = {
        key: value
        for key, value in item.items()
        if key
        not in {
            "query",
            "question",
            "instruction",
            "prompt",
            "input",
            "messages",
            "ground_truth",
            "answer",
            "target",
            "output",
            "label",
            "expected",
            "reference",
            "gold",
            "task",
            "task_name",
            "subtask",
            "subset",
            "category",
            "file_name",
            "matching_strategy",
            "metric",
            "answer_type",
            "id",
            "sample_id",
            "uuid",
        }
    }

    return SmilesExample(
        id=str(example_id),
        question=question,
        answer=answer,
        config=config,
        task=task or config,
        matching_strategy=matching_strategy,
        metadata=metadata,
    )


def _load_config_dataset(load_dataset, download_config_cls, config: str, split: str, *, local_only: bool):
    kwargs: dict[str, Any] = {"path": "OpenMol/ChemCoTBench", "name": config}
    if local_only:
        kwargs["download_config"] = download_config_cls(local_files_only=True)

    split_candidates = [split, "test", "validation", "dev", "train"]
    seen: set[str] = set()
    split_candidates = [name for name in split_candidates if not (name in seen or seen.add(name))]

    last_exc: Exception | None = None
    for split_name in split_candidates:
        try:
            return load_dataset(split=split_name, **kwargs)
        except Exception as exc:
            last_exc = exc

    try:
        dataset_dict = load_dataset(**kwargs)
        if hasattr(dataset_dict, "keys"):
            for split_name in split_candidates:
                if split_name in dataset_dict:
                    return dataset_dict[split_name]
            for split_name in dataset_dict.keys():
                return dataset_dict[split_name]
    except Exception as exc:
        last_exc = exc

    assert last_exc is not None
    raise last_exc


def load_smiles(
    split: str = "test",
    limit: Optional[int] = None,
    random_sample: bool = False,
    seed: int = 42,
    configs: Sequence[str] | None = None,
) -> list[SmilesExample]:
    """
    Load the SMILES benchmark from Chem-CoT-Bench and normalize examples.
    """
    try:
        from datasets import DownloadConfig, load_dataset
    except ImportError as e:
        raise RuntimeError(
            "Missing dependency `datasets`. Install with: pip install datasets"
        ) from e

    requested_configs = tuple(configs) if configs is not None else DEFAULT_SMILES_CONFIGS
    if not requested_configs:
        raise ValueError("At least one SMILES config must be requested.")

    print(
        "Loading SMILES benchmark dataset "
        f"(configs={','.join(requested_configs)}, split={split}, limit={limit}, random_sample={random_sample})..."
    )

    offline_only = _datasets_offline_enabled()

    def _load_all(local_only: bool) -> list[SmilesExample]:
        combined: list[SmilesExample] = []
        for config in requested_configs:
            ds = _load_config_dataset(load_dataset, DownloadConfig, config, split, local_only=local_only)
            for idx in range(len(ds)):
                normalized = _normalize_item(dict(ds[idx]), idx, config)
                if normalized is not None:
                    combined.append(normalized)
        return combined

    try:
        raw_examples = _load_all(local_only=offline_only)
    except Exception as exc:
        if offline_only or not _is_hf_connection_error(exc):
            raise
        print("  HuggingFace dataset lookup failed; retrying from local cache only.")
        raw_examples = _load_all(local_only=True)

    if limit is not None and limit > 0 and len(raw_examples) > limit:
        if random_sample:
            rng = random.Random(seed)
            raw_examples = rng.sample(raw_examples, limit)
        else:
            raw_examples = raw_examples[:limit]

    print(f"Loaded {len(raw_examples)} SMILES examples")
    return raw_examples
