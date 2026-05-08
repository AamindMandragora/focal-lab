"""Static SMILES prompt/grammar tasks copied from the CARS benchmark."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

SMILES_CLASSES: tuple[str, ...] = ("acrylates", "chain_extenders", "isocyanates")
DATA_DIR = Path(
    os.environ.get("SMILES_DATA_DIR", str(Path(__file__).resolve().parent / "data"))
).expanduser()
GRAMMAR_DIR = Path(
    os.environ.get("SMILES_GRAMMAR_DIR", str(Path(__file__).resolve().parents[2] / "grammars"))
).expanduser()


def _normalize_classes(classes: Sequence[str] | str | None) -> list[str]:
    if classes is None:
        return list(SMILES_CLASSES)
    if isinstance(classes, str):
        raw = [part.strip() for part in classes.split(",")]
    else:
        raw = [str(part).strip() for part in classes]
    selected = [part for part in raw if part]
    unknown = sorted(set(selected) - set(SMILES_CLASSES))
    if unknown:
        raise ValueError(f"Unknown SMILES class(es): {unknown}. Expected one of {SMILES_CLASSES}.")
    return selected


def extract_prompt_exemplars(prompt: str) -> list[str]:
    exemplars: list[str] = []
    for line in prompt.splitlines():
        line = line.strip()
        if line.startswith("Molecule:"):
            value = line.split(":", 1)[1].strip()
            if value:
                exemplars.append(value)
    return exemplars


@lru_cache(maxsize=None)
def get_smiles_task(class_name: str) -> Dict[str, Any]:
    class_name = class_name.strip()
    if class_name not in SMILES_CLASSES:
        raise ValueError(f"Unknown SMILES class: {class_name}")
    grammar_path = GRAMMAR_DIR / f"smiles_{class_name}.lark"
    prompt_path = DATA_DIR / f"{class_name}.txt"
    grammar_text = grammar_path.read_text()
    prompt = prompt_path.read_text()
    return {
        "class_name": class_name,
        "question": class_name,
        "prompt": prompt,
        "grammar_path": grammar_path,
        "grammar_text": grammar_text,
        "prompt_path": prompt_path,
        "prompt_exemplars": extract_prompt_exemplars(prompt),
    }


def load_smiles(
    classes: Sequence[str] | str | None = None,
    samples_per_class: int = 1,
) -> List[Dict[str, Any]]:
    """Return repeated class tasks for feedback evaluation attempts."""
    if samples_per_class < 1:
        raise ValueError("samples_per_class must be >= 1")
    rows: list[dict[str, Any]] = []
    for class_name in _normalize_classes(classes):
        task = get_smiles_task(class_name)
        for attempt_index in range(samples_per_class):
            row = dict(task)
            row["attempt_index"] = attempt_index
            rows.append(row)
    return rows
