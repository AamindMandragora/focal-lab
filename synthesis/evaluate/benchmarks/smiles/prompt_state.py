"""Incremental SMILES prompt context for multi-sample evaluation runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

_MAX_SUFFIX_CHARS = 45000
_MAX_DUPLICATE_RESPONSE_CHARS = 1500
DUPLICATE_RESPONSE_PREFIX = "[response] "
BAD_MISTAKES_LINE = "Below are past mistakes — do not repeat them."

RecordOutcome = Literal["empty", "exemplar", "good", "bad", "duplicate"]


def normalize_duplicate_response(raw: str | None) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    if len(text) > _MAX_DUPLICATE_RESPONSE_CHARS:
        return text[:_MAX_DUPLICATE_RESPONSE_CHARS] + "..."
    return text


def _render_bad_entry_lines(entry: str) -> list[str]:
    if entry.startswith(DUPLICATE_RESPONSE_PREFIX):
        body = entry[len(DUPLICATE_RESPONSE_PREFIX) :]
        return ["Response:", body]
    return [entry]


def format_good_bad_feedback_suffix(
    good_results: Sequence[str],
    bad_results: Sequence[str],
    *,
    trailing_reasoning: bool = False,
) -> str:
    """Build good/bad attempt history before the next generation slot."""
    if not good_results and not bad_results:
        return ""
    lines: list[str] = []
    if good_results:
        lines.append("Good results:")
        lines.extend(f"SMILES: {smiles}" for smiles in good_results)
    if bad_results:
        lines.append("Bad results:")
        lines.append(BAD_MISTAKES_LINE)
        lines.append("")
        for entry in bad_results:
            lines.extend(_render_bad_entry_lines(entry))
    text = "\n".join(lines)
    if trailing_reasoning:
        text += "\nReasoning:"
    return "\n" + text + "\n"


def format_attempt_suffix(
    good_results: Sequence[str],
    bad_results: Sequence[str],
) -> str:
    """Build the good/bad SMILES suffix appended before the next attempt."""
    return format_good_bad_feedback_suffix(
        good_results,
        bad_results,
        trailing_reasoning=True,
    )


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

    def _append_duplicate_response(self, cleaned: str, raw_response: str | None) -> None:
        if cleaned not in self.bad_results:
            return
        normalized = normalize_duplicate_response(raw_response)
        if normalized:
            marker_entry = f"{DUPLICATE_RESPONSE_PREFIX}{normalized}"
            if marker_entry not in self.bad_results:
                self.bad_results.append(marker_entry)
        repeat_n = sum(1 for entry in self.bad_results if entry.startswith("[repeat ")) + 1
        self.bad_results.append(f"[repeat {repeat_n}]")

    def _record_duplicate(self, cleaned: str, raw_response: str | None) -> RecordOutcome:
        if cleaned not in self.bad_results:
            self.bad_results.append(cleaned)
        else:
            self._append_duplicate_response(cleaned, raw_response)
        return "duplicate"

    def record_attempt(
        self,
        smiles: str,
        eval_row: dict[str, Any] | None,
        *,
        raw_response: str | None = None,
    ) -> RecordOutcome:
        cleaned = str(smiles or "").strip()
        if not cleaned:
            invalid_marker = "(invalid)"
            if invalid_marker not in self.bad_results:
                self.bad_results.append(invalid_marker)
            self.seen.add(invalid_marker)
            return "empty"

        row = eval_row or {}
        is_exemplar = bool(row.get("is_prompt_exemplar")) or cleaned in self.prompt_exemplars

        if cleaned in self.good_results:
            return self._record_duplicate(cleaned, raw_response)

        if is_exemplar:
            if cleaned in self.bad_results:
                return self._record_duplicate(cleaned, raw_response)
            if cleaned not in self.bad_results:
                self.bad_results.append(cleaned)
            self.seen.add(cleaned)
            return "exemplar"

        if cleaned in self.seen:
            return self._record_duplicate(cleaned, raw_response)

        is_good = bool(row.get("unique_valid_candidate"))
        if is_good:
            self.good_results.append(cleaned)
            self.seen.add(cleaned)
            return "good"

        # Any non-good attempt (syntax-invalid, wrong class, or exemplar copy) goes in Bad results.
        if cleaned not in self.bad_results:
            self.bad_results.append(cleaned)
        self.seen.add(cleaned)
        if is_exemplar:
            return "exemplar"
        return "bad"

    def build_suffix(self) -> str:
        return format_attempt_suffix(self.good_results, self.bad_results)

    def apply_to_example(self, example: dict[str, Any]) -> None:
        from synthesis.evaluate.benchmarks.smiles.native_prompt import (
            render_native_smiles_prompt_with_feedback,
        )

        class_name = str(example.get("class_name", "smiles"))
        tier = example.get("prompt_tier", 1)
        example["prompt"] = render_native_smiles_prompt_with_feedback(
            class_name,
            good_results=self.good_results,
            bad_results=self.bad_results,
            tier=tier,
        )
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
    *,
    raw_response: str | None = None,
) -> dict[str, Any] | None:
    class_name = str(example.get("class_name", "smiles"))
    outcome = states[class_name].record_attempt(smiles, eval_row, raw_response=raw_response)
    updated = dict(eval_row or {})
    updated["prompt_record_outcome"] = outcome
    updated["novel_valid"] = outcome == "good"
    return updated
