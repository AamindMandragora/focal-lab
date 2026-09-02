"""Suite hooks: twin accuracy + helper micros.

Callers: orchestrator via BabysitterHooks; production_hooks; focal_injectors.
API: TWINS / SuiteOutcome / classify_tier_a — no data files.
User instruction: babysitter-implement (suite …); gsm→Crane, spider→IterGen,
                  smiles→CARS.
"""

from __future__ import annotations

from dataclasses import dataclass


TWINS = {
    "gsm": "Crane",
    "spider": "IterGen",
    "smiles": "CARS",
}


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
