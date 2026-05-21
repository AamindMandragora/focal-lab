"""Incremental SMILES prompt context for multi-sample evaluation runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

_MAX_SUFFIX_CHARS = 45000

RecordOutcome = Literal["empty", "exemplar", "good", "bad", "duplicate"]


def strip_trailing_molecule_slot(prompt: str) -> str:
    text = prompt.rstrip()
    for suffix in ("Molecule: <<", "Molecule:", "SMILES:", "Reasoning:", "<<"):
        while text.endswith(suffix):
            text = text[: -len(suffix)].rstrip()
    return text


def _cap_suffix(suffix: str, *, max_chars: int = _MAX_SUFFIX_CHARS) -> str:
    if len(suffix) <= max_chars:
        return suffix
    lines = suffix.split("\n")
    while len("\n".join(lines)) > max_chars and len(lines) > 1:
        lines.pop(0)
    return "\n".join(lines)


@dataclass
class SmilesPromptState:
    """Tracks prior attempts so prompts can list good and bad molecules."""

    prompt_exemplars: set[str] = field(default_factory=set)
    seen: set[str] = field(default_factory=set)
    good_results: list[str] = field(default_factory=list)
    bad_results: list[str] = field(default_factory=list)

    def __init__(self, prompt_exemplars: Sequence[str] | None = None) -> None:
        exemplars = {str(value).strip() for value in (prompt_exemplars or []) if str(value).strip()}
        self.prompt_exemplars = exemplars
        self.seen = set(exemplars)
        self.good_results = []
        self.bad_results = []

    def record_attempt(self, smiles: str, eval_row: dict[str, Any] | None) -> RecordOutcome:
        cleaned = str(smiles or "").strip()
        if not cleaned:
            return "empty"

        row = eval_row or {}
        if row.get("is_prompt_exemplar") or cleaned in self.prompt_exemplars:
            self.seen.add(cleaned)
            return "exemplar"

        if cleaned in self.good_results:
            self.seen.add(cleaned)
            return "duplicate"

        is_good = bool(row.get("unique_valid_candidate"))
        if is_good:
            self.good_results.append(cleaned)
            self.seen.add(cleaned)
            return "good"

        if cleaned not in self.bad_results:
            self.bad_results.append(cleaned)
        self.seen.add(cleaned)
        return "bad"

    def build_suffix(self) -> str:
        if not self.good_results and not self.bad_results:
            return ""
        lines: list[str] = []
        if self.good_results:
            lines.append("Good results:")
            lines.extend(f"SMILES: {smiles}" for smiles in self.good_results)
        if self.bad_results:
            lines.append("Bad results:")
            lines.extend(f"SMILES: {smiles}" for smiles in self.bad_results)
        lines.append("Reasoning:")
        return "\n" + "\n".join(lines) + "\n"

    def apply_to_example(self, example: dict[str, Any]) -> None:
        base_key = "_smiles_base_prompt"
        if base_key not in example:
            example[base_key] = strip_trailing_molecule_slot(str(example.get("prompt", "")))
        suffix = _cap_suffix(self.build_suffix())
        example["prompt"] = example[base_key] + suffix
        example["smiles_good_results"] = list(self.good_results)
        example["smiles_bad_results"] = list(self.bad_results)


def init_prompt_states(dataset: Sequence[dict[str, Any]]) -> dict[str, SmilesPromptState]:
    states: dict[str, SmilesPromptState] = {}
    for row in dataset:
        class_name = str(row.get("class_name", "smiles"))
        if class_name not in states:
            states[class_name] = SmilesPromptState(row.get("prompt_exemplars", []))
    return states


def apply_prompt_state(example: dict[str, Any], states: dict[str, SmilesPromptState]) -> None:
    class_name = str(example.get("class_name", "smiles"))
    states[class_name].apply_to_example(example)


def record_prompt_result(
    example: dict[str, Any],
    states: dict[str, SmilesPromptState],
    smiles: str,
    eval_row: dict[str, Any] | None,
) -> dict[str, Any] | None:
    class_name = str(example.get("class_name", "smiles"))
    outcome = states[class_name].record_attempt(smiles, eval_row)
    updated = dict(eval_row or {})
    updated["prompt_record_outcome"] = outcome
    updated["novel_valid"] = outcome == "good"
    return updated
