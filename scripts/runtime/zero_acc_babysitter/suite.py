"""Suite hooks: twin accuracy + helper micros.

Callers: orchestrator via BabysitterHooks; focal_injectors.
API: SuiteOutcome / classify_tier_a — no data files.
User instruction: babysitter-implement (suite …).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SuiteOutcome:
    twin_accuracy: float
    helper_failures: list[str]


def classify_tier_a(outcome: SuiteOutcome) -> str:
    if outcome.twin_accuracy <= 0.0:
        return "harness"
    if outcome.helper_failures:
        return "helper"
    return "strategy_miss"
