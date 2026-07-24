"""Log watcher: extract finished-attempt Accuracy / MEMORY_OPS signals.

Callers: __main__.py babysitter loop; accuracy helpers.
API: build_watch_event / detect_memory_ops / is_attempt_finished — no data files.
User instruction: babysitter-implement (watcher, suite, Cloud hooks, …).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from scripts.runtime.zero_acc_babysitter.accuracy import (
    attempt_block,
    parse_finished_attempt_accuracy,
)
from scripts.runtime.zero_acc_babysitter.constants import (
    ATTEMPT_START_RE,
    MEMORY_OPS_SIGNATURES,
    SYNTHESIS_FINISH_RE,
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


def _pid_file_exited(pid_file: Path | None) -> bool:
    """True when a Phase-2 pid file exists but the process is gone."""
    if pid_file is None or not pid_file.is_file():
        return False
    try:
        raw = pid_file.read_text(encoding="utf-8").strip()
        pid = int(raw.split()[0])
    except (OSError, ValueError, IndexError):
        return False
    try:
        Path(f"/proc/{pid}").stat()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    # /proc exists only on Linux; on macOS fall back to os.kill(pid, 0).
    import os

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    except OSError:
        return False
    return False


def is_attempt_finished(
    text: str,
    attempt_index: int,
    *,
    pid_file: Path | None = None,
) -> bool:
    """Attempt N is finished iff a later attempt started, a finish marker
    appears after N's header, or the cell pid file's process has exited.
    """
    starts = list(ATTEMPT_START_RE.finditer(text))
    match_i = None
    for i, match in enumerate(starts):
        if int(match.group("number")) == attempt_index:
            match_i = i
            break
    if match_i is None:
        return False
    if match_i + 1 < len(starts):
        return True
    start_pos = starts[match_i].start()
    for fin in SYNTHESIS_FINISH_RE.finditer(text):
        if fin.start() >= start_pos:
            return True
    if _pid_file_exited(pid_file):
        return True
    return False


def build_watch_event(text: str, attempt_index: int, *, finished: bool) -> WatchEvent:
    parsed = parse_finished_attempt_accuracy(text, attempt_index, finished=finished)
    block = attempt_block(text, attempt_index) or ""
    return WatchEvent(
        attempt_index=attempt_index,
        finished=finished,
        accuracy_pct=parsed.accuracy_pct if finished else None,
        # Scope MEMORY_OPS to this attempt's slice only (B8).
        memory_ops=detect_memory_ops(block) if finished else False,
    )
