"""Native CARS-style SMILES prompts (``acrylates.txt`` layout)."""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

from synthesis.evaluate.benchmarks.smiles.dataset import DATA_DIR, SMILES_CLASSES, extract_prompt_exemplars
from synthesis.evaluate.prompt_tiers import PromptTier


def _smiles_class_label(class_name: str) -> str:
    return str(class_name).replace("_", " ")


def native_smiles_prompt_header(class_name: str, *, tier: PromptTier = 1) -> str:
    """Tier-specific instruction block prepended to native exemplar molecules."""
    class_name = class_name.strip()
    if class_name not in SMILES_CLASSES:
        raise ValueError(f"Unknown SMILES class: {class_name}")
    label = _smiles_class_label(class_name)
    response_line = (
        "Your response must be a single SMILES string using SMILES notation only "
        "(not IUPAC names, systematic names, or prose). Output nothing else."
    )
    if tier == 1:
        return (
            f"You are an expert in chemistry. Your task is to generate one new, valid "
            f"{label} molecule in SMILES format.\n\n{response_line}"
        )
    return (
        f"You are an expert in chemistry. You are given several examples of {label} "
        f"molecules in SMILES format. Look at the given molecules, identify patterns, "
        f"then generate one new, valid {label} molecule in SMILES format.\n\n{response_line}"
    )


@lru_cache(maxsize=None)
def full_prompt_exemplars(class_name: str) -> tuple[str, ...]:
    """All ``Molecule:`` exemplars from the static class prompt file."""
    class_name = class_name.strip()
    if class_name not in SMILES_CLASSES:
        raise ValueError(f"Unknown SMILES class: {class_name}")
    prompt_path = DATA_DIR / f"{class_name}.txt"
    return tuple(extract_prompt_exemplars(prompt_path.read_text(), limit=None))


def merge_smiles_exemplars(*groups: Sequence[str]) -> list[str]:
    """Dedupe exemplar SMILES while preserving order."""
    merged: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for value in group:
            cleaned = str(value).strip()
            if not cleaned or cleaned in seen:
                continue
            merged.append(cleaned)
            seen.add(cleaned)
    return merged


def render_native_smiles_prompt(
    class_name: str,
    exemplars: Sequence[str],
    *,
    tier: PromptTier = 1,
) -> str:
    """Render a CARS-style prompt ending with an empty ``Molecule:`` generation slot."""
    header = native_smiles_prompt_header(class_name, tier=tier)
    lines = [header, ""]
    for smiles in merge_smiles_exemplars(exemplars):
        lines.append(f"Molecule: {smiles}")
    lines.append("Molecule: ")
    return "\n".join(lines)


def format_native_feedback_suffix(
    good_results: Sequence[str],
    bad_results: Sequence[str],
) -> str:
    """Good/bad attempt history appended before the generation slot (no CoT tail)."""
    from synthesis.evaluate.benchmarks.smiles.prompt_state import format_good_bad_feedback_suffix

    return format_good_bad_feedback_suffix(good_results, bad_results)


def render_native_smiles_prompt_with_feedback(
    class_name: str,
    *,
    good_results: Sequence[str] = (),
    bad_results: Sequence[str] = (),
    tier: PromptTier = 1,
) -> str:
    """Static native exemplars plus optional good/bad feedback before the empty slot."""
    from synthesis.evaluate.benchmarks.smiles.prompt_state import _cap_suffix

    base = render_native_smiles_prompt(
        class_name,
        full_prompt_exemplars(class_name),
        tier=tier,
    ).rstrip()
    if base.endswith("Molecule:"):
        base = base[: -len("Molecule:")].rstrip()
    suffix = _cap_suffix(format_native_feedback_suffix(good_results, bad_results))
    if suffix:
        return f"{base}{suffix}Molecule: "
    return f"{base}\nMolecule: "
