"""Hardened pre-PR smoke gate.

Callers: orchestrator via BabysitterHooks.run_smoke.
API: SmokeGate.evaluate(incident) -> SmokeReport; no data files.
User instruction: babysitter-implement (… smoke gate …).
"""

from __future__ import annotations

from dataclasses import dataclass

from scripts.runtime.zero_acc_babysitter.constants import PathKind, SMOKE_MODEL
from scripts.runtime.zero_acc_babysitter.persistence import IncidentRecord


@dataclass
class SmokeReport:
    passed: bool
    command: str
    metrics: dict[str, float]


class SmokeGate:
    def __init__(self, decide) -> None:
        self._decide = decide

    def evaluate(self, incident: IncidentRecord) -> SmokeReport:
        passed = bool(self._decide(incident))
        command = (
            f"smoke model={SMOKE_MODEL} kind={incident.path_kind} "
            f"iters<=3 tiny-N cell={incident.cell_id}"
        )
        return SmokeReport(
            passed=passed,
            command=command,
            metrics={"acc_proxy": 1.0 if passed else 0.0},
        )


def path_requires_twin(path_kind: str) -> bool:
    return path_kind in {PathKind.HARNESS.value, PathKind.TELEMETRY.value}
