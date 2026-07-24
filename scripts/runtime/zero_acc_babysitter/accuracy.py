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
    # A crashed run appended to the same log restarts attempt numbering, so
    # the same attempt index can appear more than once; the last one is the
    # run that actually finished.
    starts = list(ATTEMPT_START_RE.finditer(text))
    for i in range(len(starts) - 1, -1, -1):
        if int(starts[i].group("number")) != attempt_index:
            continue
        end = starts[i + 1].start() if i + 1 < len(starts) else len(text)
        return text[starts[i].start() : end]
    return None


def _strip_claude_stream_noise(block: str) -> str:
    """Drop [claude-stream] lines and their indented continuations (B12).

    Blank lines end the continuation so a later real indented
    ``Accuracy:`` line is not swallowed.
    """
    out: list[str] = []
    skipping_stream = False
    for line in block.splitlines(keepends=True):
        if line.startswith("[claude-stream]"):
            skipping_stream = True
            continue
        if skipping_stream:
            if not line.strip():
                skipping_stream = False
                out.append(line)
                continue
            if line[:1] in (" ", "	"):
                continue
            skipping_stream = False
        out.append(line)
    return "".join(out)


def parse_accuracy_in_block(block: str) -> float | None:
    """Return the eval Accuracy percent, ignoring Required-thresholds lines."""
    cleaned = _strip_claude_stream_noise(block)
    cleaned = re.sub(
        r"Required thresholds:.*?(?=\n\S|\Z)",
        "",
        cleaned,
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
