"""Default benchmark plugin hooks shared by GSM-Symbolic and Spider eval_logic."""

from __future__ import annotations

from typing import Any


def uses_hidden_chunks() -> bool:
    return False


def example_syntax_pass_from_segments(
    all_valid_syntax: bool,
    segments: list[tuple[str, bool]],
    used_hidden_chunk: bool,
    aux: dict[str, Any] | None,
) -> bool:
    return bool(segments) and all_valid_syntax


def accuracy_applicable_always(aux: dict[str, Any] | None) -> bool:
    return True


def accuracy_upper_bound_with_remaining(
    num_correct: int,
    remaining: int,
    num_accuracy_examples: int,
    total_planned_examples: int,
) -> float:
    return (num_correct + remaining) / max(1, total_planned_examples)


def final_accuracy_denominator_all_examples(
    num_examples: int,
    num_accuracy_examples: int,
) -> int:
    return num_examples


def invalid_outputs_excluded_none(
    num_examples: int,
    num_accuracy_examples: int,
) -> int:
    return 0


def accuracy_definition_standard() -> str:
    return "correct_examples_over_all_examples"
