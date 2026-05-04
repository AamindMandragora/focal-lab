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
import random
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Sequence

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


def load_gsm_symbolic_mixed(
    per_config: int = 10,
    split: str = 'test',
    seed: Optional[int] = 42,
) -> list[dict[str, Any]]:
    """
    Load a mixed sample from all three GSM-Symbolic configs (main, p1, p2).

    Each example is tagged with a 'config' field so per-difficulty metrics
    can be computed downstream.
    """
    rows: list[dict[str, Any]] = []
    for config in ('main', 'p1', 'p2'):
        chunk = load_gsm_symbolic(
            config=config, split=split, limit=per_config,
            random_sample=True, seed=seed,
        )
        for ex in chunk:
            ex['config'] = config
        rows.extend(chunk)
    print(f'Mixed dataset: {len(rows)} examples ({per_config} per config)')
    return rows


def load_gsm_from_crane_folder(
    crane_dir: Path | str | None = None,
    limit: Optional[int] = None,
    indices: Optional[Sequence[int]] = None,
) -> list[dict[str, Any]]:
    """
    Load GSM-Symbolic examples directly from CRANE's local JSON files.

    Args:
        crane_dir: Path to folder of JSON files (defaults to DEFAULT_CRANE_GSM_DIR).
        limit: Optional limit on number of examples.
        indices: Optional sorted-folder indices to select before applying limit.

    Returns:
        List of example dicts with the same fields the evaluator expects:
        question, answer, variable_types, question_parsed, etc.
    """
    crane_dir = Path(crane_dir) if crane_dir is not None else DEFAULT_CRANE_GSM_DIR
    if not crane_dir.exists():
        raise FileNotFoundError(f"CRANE GSM folder not found: {crane_dir}")

    paths = sorted(crane_dir.glob('*.json'))
    if indices is not None:
        selected_paths = []
        for idx in indices:
            if idx < 0 or idx >= len(paths):
                raise IndexError(
                    f"GSM CRANE index {idx} is out of range for {len(paths)} files in {crane_dir}"
                )
            selected_paths.append(paths[idx])
        paths = selected_paths

    rows: list[dict[str, Any]] = []
    for original_index, path in enumerate(paths):
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        rows.append({
            'question': payload.get('question', ''),
            'answer': payload.get('answer', ''),
            'variable_types': payload.get('variable_types') or {},
            'question_parsed': payload.get('question_parsed') or '',
            'question_annotated': payload.get('question_annotated') or '',
            'answer_parsed': payload.get('answer_parsed') or '',
            'id_orig': payload.get('id_orig'),
            'id_shuffled': payload.get('id_shuffled'),
            'crane_source_file': str(path),
            'crane_source_index': (
                indices[original_index] if indices is not None else original_index
            ),
        })

    if limit is not None and limit > 0:
        rows = rows[:limit]

    print(f'Loaded {len(rows)} examples from CRANE folder {crane_dir}')
    return rows


def make_gsm_train_eval_split(
    total_examples: int,
    train_fraction: float = 0.5,
    seed: int = 123,
) -> dict[str, Any]:
    """Create a deterministic complementary train/eval split over sorted examples."""
    if total_examples <= 0:
        raise ValueError("total_examples must be positive")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")

    train_size = int(round(total_examples * train_fraction))
    train_size = max(1, min(total_examples - 1, train_size))
    rng = random.Random(seed)
    train_indices = sorted(rng.sample(range(total_examples), train_size))
    train_set = set(train_indices)
    eval_indices = [idx for idx in range(total_examples) if idx not in train_set]

    return {
        "seed": seed,
        "train_fraction": train_fraction,
        "total_examples": total_examples,
        "train_indices": train_indices,
        "eval_indices": eval_indices,
        "train_size": len(train_indices),
        "eval_size": len(eval_indices),
    }


def write_gsm_train_eval_split(
    output_path: Path | str,
    crane_dir: Path | str | None = None,
    train_fraction: float = 0.5,
    seed: int = 123,
) -> dict[str, Any]:
    """Write a deterministic train/eval split manifest for the local CRANE GSM folder."""
    crane_dir = Path(crane_dir) if crane_dir is not None else DEFAULT_CRANE_GSM_DIR
    total_examples = len(sorted(crane_dir.glob('*.json')))
    split = make_gsm_train_eval_split(
        total_examples=total_examples,
        train_fraction=train_fraction,
        seed=seed,
    )
    split["crane_dir"] = str(crane_dir)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(split, indent=2))
    return split


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
