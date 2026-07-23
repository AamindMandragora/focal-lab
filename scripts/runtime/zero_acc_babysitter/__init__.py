"""Zero-acc babysitter package (Cursor CLI repair; Cloud Agents unused).

Callers: tests, `python -m scripts.runtime.zero_acc_babysitter`, focal mock suite.
API: exports CellState, PathKind, LocalBabysitterSim, CursorCliClient, NullCloudClient.
Data schemas: none (re-exports only).
User instruction: "Replace/extend NullCloudClient with a real CursorCliClient".
"""

from scripts.runtime.zero_acc_babysitter.cloud import (
    DEFAULT_CURSOR_AGENT_MODEL,
    CursorCliClient,
    NullCloudClient,
)
from scripts.runtime.zero_acc_babysitter.constants import CellState, PathKind
from scripts.runtime.zero_acc_babysitter.local_sim import LocalBabysitterSim

__all__ = [
    "CellState",
    "PathKind",
    "LocalBabysitterSim",
    "CursorCliClient",
    "DEFAULT_CURSOR_AGENT_MODEL",
    "NullCloudClient",
]
