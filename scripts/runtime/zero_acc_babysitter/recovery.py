"""Recovery actions after a successful smoke+merge.

Callers: orchestrator recovery phases.
API: helper_first_use_k() — no data files.
User instruction: babysitter-implement (… memory-then-accuracy, TELEMETRY_FAIL, tiers …).
"""

from __future__ import annotations

from typing import Sequence


def helper_first_use_k(
    helper_trace_by_attempt: dict[int, list[str]],
    failing_helpers: Sequence[str],
    fallback_attempt: int,
) -> int:
    failing = set(failing_helpers)
    hits = [
        int(idx)
        for idx, used in helper_trace_by_attempt.items()
        if failing.intersection(used)
    ]
    return min(hits) if hits else fallback_attempt
