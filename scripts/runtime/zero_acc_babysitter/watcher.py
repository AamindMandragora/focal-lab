"""Log watcher: extract finished-attempt Accuracy / MEMORY_OPS signals.

Callers: __main__.py babysitter loop; accuracy helpers.
API: build_watch_event / detect_memory_ops — no data files.
User instruction: babysitter-implement (watcher, suite, Cloud hooks, …).
"""

from __future__ import annotations

from dataclasses import dataclass

from scripts.runtime.zero_acc_babysitter.accuracy import parse_finished_attempt_accuracy
from scripts.runtime.zero_acc_babysitter.constants import (
    ATTEMPT_START_RE,
    MEMORY_OPS_SIGNATURES,
)


@dataclass(frozen=True)
class WatchEvent:
    attempt_index: int
    finished: bool
    accuracy_pct: float | None
    memory_ops: bool


def detect_memory_ops(text: str) -> bool:
    lowered = text.lower()
    return any(sig in lowered for sig in MEMORY_OPS_SIGNATURES)


def latest_attempt_index(text: str) -> int | None:
    matches = list(ATTEMPT_START_RE.finditer(text))
    if not matches:
        return None
    return int(matches[-1].group("number"))


def build_watch_event(text: str, attempt_index: int, *, finished: bool) -> WatchEvent:
    parsed = parse_finished_attempt_accuracy(text, attempt_index, finished=finished)
    return WatchEvent(
        attempt_index=attempt_index,
        finished=finished,
        accuracy_pct=parsed.accuracy_pct if finished else None,
        memory_ops=detect_memory_ops(text),
    )
