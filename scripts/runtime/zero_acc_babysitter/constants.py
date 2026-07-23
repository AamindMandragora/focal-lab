"""Locked constants for the zero-acc Cloud babysitter."""

from __future__ import annotations

import re
from enum import Enum

MAX_CLOUD_ATTEMPTS = 30
TIER_B_ACC_PCT = 15.0
SMOKE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MERGE_BRANCH = "synthesis-snapshot-20260622"

# Plan regex required EOL; live failure lines append " (min: ...)".
# Capture group matches the plan; trailing content after % is allowed.
ACCURACY_LINE_RE = re.compile(
    r"^\s*Accuracy:\s*([0-9]+(?:\.[0-9]+)?)%\s*",
    re.MULTILINE,
)
ATTEMPT_START_RE = re.compile(
    r"^Attempt\s+(?P<number>\d+)\s*/",
    re.MULTILINE,
)
MEMORY_OPS_SIGNATURES = (
    "cuda out of memory",
    "torch.OutOfMemoryError",
    "out of memory",
    "oom-kill",
)


class CellState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED_FOR_SUITE = "paused_for_suite"
    INCIDENT_ACTIVE = "incident_active"
    BLOCKED_NEEDS_HUMAN = "blocked_needs_human"
    ACCEPTED = "accepted"
    FAILED_EXHAUSTED = "failed_exhausted"


class PathKind(str, Enum):
    HARNESS = "harness"
    HELPER = "helper"
    TELEMETRY = "telemetry"
    MEMORY = "memory"
    TIER_B_HELPER = "tier_b_helper"


class IncidentPhase(str, Enum):
    OPEN = "open"
    KILLED = "killed"
    CLOUD = "cloud"
    SMOKE = "smoke"
    MERGE = "merge"
    RECOVERY = "recovery"
    CLOSED = "closed"
