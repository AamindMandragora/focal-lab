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


def normalize_smiles_classes(
    classes: Sequence[str] | str | None,
    *,
    dedupe: bool = False,
    require_non_empty: bool = False,
) -> list[str]:
    """Normalize SMILES class names with validation against known classes."""
    if classes is None:
        selected = list(SMILES_CLASSES)
    elif isinstance(classes, str):
        raw = [part.strip() for part in classes.split(",")]
        selected = [part for part in raw if part]
    else:
        selected = [str(part).strip() for part in classes if str(part).strip()]

    unknown = sorted(set(selected) - set(SMILES_CLASSES))
    if unknown:
        raise ValueError(f"Unknown SMILES class(es): {unknown}. Expected one of {SMILES_CLASSES}.")

    if dedupe:
        deduped: list[str] = []
        seen: set[str] = set()
        for class_name in selected:
            if class_name in seen:
                continue
            deduped.append(class_name)
            seen.add(class_name)
        selected = deduped

    if require_non_empty and not selected:
        raise ValueError("At least one SMILES class is required.")

    return selected


def extract_prompt_exemplars(prompt: str) -> list[str]:
    exemplars: list[str] = []
    for line in prompt.splitlines():
        line = line.strip()
        if line.startswith("Molecule:") or line.startswith("SMILES:"):
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
    properties = prompt.split("Molecule:")[0].split("SMILES:")[0].strip()
    if properties.startswith("Properties:"):
        properties = properties.split(":", 1)[1].strip()
    return {
        "class_name": class_name,
        "question": class_name,
        "prompt": prompt,
        "smiles_properties": properties,
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
    for class_name in normalize_smiles_classes(classes):
        task = get_smiles_task(class_name)
        for attempt_index in range(samples_per_class):
            row = dict(task)
            row["attempt_index"] = attempt_index
            rows.append(row)
    return rows
