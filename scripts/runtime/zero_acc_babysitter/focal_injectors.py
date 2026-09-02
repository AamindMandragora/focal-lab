"""Focal throwaway injectors (real Cloud later; stubs for Phase 1).

Callers: future focal-mocks agent; not used by local sim.
API: FocalInjectorSpec / default_injectors — no real-queue writes.
User instruction: babysitter-implement (… focal injectors …); focal-mocks out of scope.
"""

from __future__ import annotations

from dataclasses import dataclass


SCENARIO_IDS = (
    "tier_a_harness",
    "tier_a_helper",
    "tier_a_miss",
    "tier_b_helper",
    "tier_b_miss",
    "memory_alone",
    "memory_then_accuracy",
    "telemetry_fail",
    "smoke_retry",
    "acc_ge_15",
    "mid_incident_resume",
)


@dataclass(frozen=True)
class FocalInjectorSpec:
    scenario_id: str
    cell_id: str
    notes: str


def default_injectors(throwaway_cell: str = "gsm-qwen25-1p5b") -> list[FocalInjectorSpec]:
    return [
        FocalInjectorSpec(s, throwaway_cell, "Phase-1 stub; real Cloud in focal-mocks")
        for s in SCENARIO_IDS
    ]
