"""Parse per-attempt Accuracy lines from synthesis logs."""

from __future__ import annotations

import re
from dataclasses import dataclass

from scripts.runtime.zero_acc_babysitter.constants import (
    ACCURACY_LINE_RE,
    ATTEMPT_START_RE,
)


@dataclass(frozen=True)
class AttemptAccuracy:
    attempt_index: int
    accuracy_pct: float | None
    finished: bool


def attempt_block(text: str, attempt_index: int) -> str | None:
    starts = list(ATTEMPT_START_RE.finditer(text))
    for i, match in enumerate(starts):
        if int(match.group("number")) != attempt_index:
            continue
        end = starts[i + 1].start() if i + 1 < len(starts) else len(text)
        return text[match.start() : end]
    return None


def parse_accuracy_in_block(block: str) -> float | None:
    """Return the eval Accuracy percent, ignoring Required-thresholds lines."""
    cleaned = re.sub(
        r"Required thresholds:.*?(?=\n\S|\Z)",
        "",
        block,
        flags=re.S,
    )
    matches = list(ACCURACY_LINE_RE.finditer(cleaned))
    if not matches:
        return None
    return float(matches[-1].group(1))


def parse_finished_attempt_accuracy(
    text: str, attempt_index: int, *, finished: bool
) -> AttemptAccuracy:
    block = attempt_block(text, attempt_index)
    if block is None:
        return AttemptAccuracy(attempt_index, None, finished)
    return AttemptAccuracy(attempt_index, parse_accuracy_in_block(block), finished)
