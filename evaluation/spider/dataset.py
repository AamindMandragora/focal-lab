"""
Spider text-to-SQL dataset loading utilities.

The loader targets HuggingFace-hosted Spider variants and normalizes examples
to the small field surface used by the synthesis evaluator.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class SpiderExample:
    """A single Spider text-to-SQL example."""

    id: str
    question: str
    query: str
    db_id: str = ""
    schema: str = ""

    @property
    def answer(self) -> str:
        return self.query


def _datasets_offline_enabled() -> bool:
    """True when dataset loading should stay offline / cache-only."""
    return any(os.environ.get(name, "").strip() in {"1", "true", "True"} for name in (
        "HF_DATASETS_OFFLINE",
        "HF_HUB_OFFLINE",
    ))


def _is_hf_connection_error(exc: Exception) -> bool:
    """Best-effort detection for HF dataset connectivity failures."""
    text = str(exc).lower()
    return any(marker in text for marker in (
        "failed to resolve",
        "name or service not known",
        "temporary failure in name resolution",
        "connection error",
        "maxretryerror",
        "httpsconnectionpool",
        "offline mode",
    ))


def _first_present(item: dict[str, Any], keys: tuple[str, ...], default: Any = "") -> Any:
    for key in keys:
        value = item.get(key)
        if value not in (None, ""):
            return value
    return default


def _format_schema(item: dict[str, Any]) -> str:
    schema = _first_present(
        item,
        (
            "schema",
            "db_schema",
            "database_schema",
            "serialized_schema",
            "schema_text",
            "context",
        ),
        "",
    )
    if schema:
        return str(schema)

    table_names = _first_present(item, ("table_names", "table_names_original"), [])
    column_names = _first_present(item, ("column_names", "column_names_original"), [])
    if not table_names and not column_names:
        return ""
    return f"tables={table_names}; columns={column_names}"


def _normalize_item(item: dict[str, Any], idx: int) -> SpiderExample:
    return SpiderExample(
        id=str(_first_present(item, ("id", "query_id", "qid"), idx)),
        question=str(_first_present(item, ("question", "utterance", "text"), "")),
        query=str(_first_present(item, ("query", "sql", "SQL"), "")),
        db_id=str(_first_present(item, ("db_id", "database_id", "db"), "")),
        schema=_format_schema(item),
    )


def load_spider(
    split: str = "test",
    limit: Optional[int] = None,
    random_sample: bool = False,
    seed: int = 42,
) -> list[SpiderExample]:
    """
    Load a Spider Text-to-SQL dataset from HuggingFace.

    The preferred dataset is ``SuperMax991/spider-text2sql`` because it includes
    schema text directly in each example. If the requested split is unavailable,
    the loader tries validation/dev/train-style split names.
    """
    try:
        from datasets import DownloadConfig, load_dataset
    except ImportError as e:
        raise RuntimeError(
            "Missing dependency `datasets`. Install with: pip install datasets"
        ) from e

    print(f"Loading Spider dataset (split={split}, limit={limit}, random_sample={random_sample})...")
    offline_only = _datasets_offline_enabled()
    split_candidates = [split, "validation", "dev", "test", "train"]
    seen: set[str] = set()
    split_candidates = [name for name in split_candidates if not (name in seen or seen.add(name))]

    def _try_load(local_only: bool):
        kwargs: dict[str, Any] = {"path": "SuperMax991/spider-text2sql"}
        if local_only:
            kwargs["download_config"] = DownloadConfig(local_files_only=True)
        last_exc: Exception | None = None
        for split_name in split_candidates:
            try:
                return load_dataset(split=split_name, **kwargs)
            except Exception as exc:
                last_exc = exc
        assert last_exc is not None
        raise last_exc

    try:
        ds = _try_load(local_only=offline_only)
    except Exception as exc:
        if offline_only or not _is_hf_connection_error(exc):
            raise
        print("  HuggingFace dataset lookup failed; retrying Spider from local cache only.")
        ds = _try_load(local_only=True)

    indices = list(range(len(ds)))
    if limit is not None and limit > 0:
        if random_sample:
            rng = random.Random(seed)
            indices = rng.sample(indices, min(limit, len(indices)))
        else:
            indices = indices[:limit]

    examples = [_normalize_item(dict(ds[i]), i) for i in indices]
    examples = [example for example in examples if example.question and example.query]
    print(f"Loaded {len(examples)} Spider examples")
    return examples
