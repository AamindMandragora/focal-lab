"""
GSM-Symbolic dataset loading utilities.

Handles loading the apple/GSM-Symbolic dataset from HuggingFace with various
configurations (main, p1, p2) and supports limiting/sampling for efficient evaluation.
Also backfills CRANE metadata such as variable_types when the local CRANE checkout
is available.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

DEFAULT_CRANE_GSM_DIR = Path(
    os.environ.get("CRANE_GSM_SYMBOLIC_DIR", "/home/aadivyar/CRANE/src/gsm_symbolic")
)


def _normalize_gsm_text(text: str) -> str:
    """Normalize question text for robust lookups across dataset variants."""
    return " ".join(str(text).split())


@lru_cache(maxsize=None)
def _load_crane_gsm_metadata(crane_dir_str: str) -> dict[str, dict[Any, dict[str, Any]]]:
    """Load CRANE GSM metadata indexed by original id and original question text."""
    crane_dir = Path(crane_dir_str)
    by_original_id: dict[int, dict[str, Any]] = {}
    by_question: dict[str, dict[str, Any]] = {}

    if not crane_dir.exists():
        return {"by_original_id": by_original_id, "by_question": by_question}

    for path in sorted(crane_dir.glob('*.json')):
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue

        metadata = {
            'variable_types': payload.get('variable_types') or {},
            'question_parsed': payload.get('question_parsed') or payload.get('question', ''),
            'question_annotated': payload.get('question_annotated') or '',
            'answer_parsed': payload.get('answer_parsed') or '',
            'crane_source_file': str(path),
            'crane_id_orig': payload.get('id_orig'),
            'crane_id_shuffled': payload.get('id_shuffled'),
        }

        original_id = payload.get('id_orig')
        if isinstance(original_id, int):
            by_original_id[original_id] = metadata

        question_text = payload.get('question')
        if question_text:
            by_question[_normalize_gsm_text(question_text)] = metadata

    return {"by_original_id": by_original_id, "by_question": by_question}


def enrich_gsm_symbolic_example(
    example: dict[str, Any],
    crane_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Attach CRANE metadata to one HF GSM-Symbolic example when available."""
    crane_dir = Path(crane_dir) if crane_dir is not None else DEFAULT_CRANE_GSM_DIR
    indices = _load_crane_gsm_metadata(str(crane_dir))

    enriched = dict(example)
    metadata = None

    original_id = enriched.get('original_id')
    if original_id is not None:
        try:
            metadata = indices['by_original_id'].get(int(original_id))
        except (TypeError, ValueError):
            metadata = None

    if metadata is None:
        original_question = enriched.get('original_question') or enriched.get('question')
        if original_question:
            metadata = indices['by_question'].get(_normalize_gsm_text(original_question))

    if metadata is None:
        return enriched

    for key, value in metadata.items():
        enriched.setdefault(key, value)

    return enriched


def load_gsm_symbolic(
    config: str = 'main',
    split: str = 'test',
    limit: Optional[int] = None,
    random_sample: bool = False,
    seed: Optional[int] = None,
):
    """
    Load GSM-Symbolic dataset from HuggingFace and enrich rows with CRANE metadata.

    Args:
        config: Dataset configuration - "main", "p1", or "p2"
                - "main": Standard difficulty
                - "p1": Perturbation level 1
                - "p2": Perturbation level 2
        split: Dataset split (usually "test")
        limit: Optional limit on number of examples to load (for efficiency)
        random_sample: If True and limit is set, randomly sample examples
                       instead of taking first N
        seed: Optional RNG seed used when random_sample=True

    Returns:
        List of dataset rows as dictionaries, enriched with CRANE metadata when available.

    Raises:
        RuntimeError: If the datasets library is not installed
        ValueError: If config is not one of the valid options
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError(
            'Missing dependency `datasets`. Install with: pip install datasets'
        ) from e

    valid_configs = ['main', 'p1', 'p2']
    if config not in valid_configs:
        raise ValueError(f'Config must be one of {valid_configs}, got: {config}')

    sample_str = ' (random sample)' if random_sample and limit else ''
    if sample_str and seed is not None:
        sample_str = f'{sample_str} (seed={seed})'
    limit_str = f' (limit={limit})' if limit else ''
    print(f'Loading GSM-Symbolic dataset (config={config}, split={split}{limit_str}{sample_str})...')

    try:
        ds = load_dataset('apple/GSM-Symbolic', name=config, split=split)
    except Exception:
        print(f"Failed to load split '{split}', trying 'test'...")
        try:
            ds = load_dataset('apple/GSM-Symbolic', name=config, split='test')
        except Exception:
            print("Failed to load 'test', trying 'train'...")
            ds = load_dataset('apple/GSM-Symbolic', name=config, split='train')

    if limit is not None and limit > 0:
        if random_sample:
            import random
            rng = random.Random(seed) if seed is not None else random
            indices = rng.sample(range(len(ds)), min(limit, len(ds)))
            ds = ds.select(indices)
        else:
            ds = ds.select(range(min(limit, len(ds))))

    rows = [enrich_gsm_symbolic_example(dict(row)) for row in ds]
    print(f'Loaded {len(rows)} examples')
    return rows
