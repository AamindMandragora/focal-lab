"""Human-readable master + per-lane logs with babysitter event markers.

Also mirrors wake/activation activity into ``activations.log`` (easy ``tail -F``).
Master keeps every marker; activations keeps triggers + actions + outcomes only.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from scripts.runtime.zero_acc_babysitter.domains import lane_name

# Poll / startup / healthy-continue noise — stay on master only.
_ACTIVATIONS_SKIP = frozenset(
    {
        "WATCHER_START",
        "CLI_PROBE_OK",
        "CLI_PROBE_FAIL",
        "CLAUDE_CODE_CLI_READY",
        "WATCH_EVENT",
        "ACC_GE_15_CONTINUE",
        "NO_WAKE",
    }
)

# Optional role tag so activations.log is skimmable (TRIGGER / ACTION / OUTCOME / SKIP).
_ACTIVATION_ROLES: dict[str, str] = {
    "TELEMETRY_FAIL": "TRIGGER",
    "TIER_A": "TRIGGER",
    "TIER_B": "TRIGGER",
    "MEMORY_OPS_START": "TRIGGER",
    "INCIDENT_RESUME": "TRIGGER",
    "INCIDENT_RESUME_STARTUP": "TRIGGER",
    "INCIDENT_ATTACH": "TRIGGER",
    "HARNESS": "ACTION",
    "HELPER": "ACTION",
    "INCIDENT_OPEN": "ACTION",
    "KILL": "ACTION",
    "CLOUD_DEBUG": "ACTION",
    "SMOKE_START": "ACTION",
    "SMOKE_FAIL": "ACTION",
    "SMOKE_PASS": "ACTION",
    "WAKE_AUTO_REPAIR_VIA_WORKTREE": "ACTION",
    "WAKE_OBSERVED_NO_AUTO_REPAIR": "ACTION",
    "CLI_AGENT_FAIL": "ACTION",
    "RESCORE_ALL": "ACTION",
    "SIBLING_REEVAL": "ACTION",
    "RESTART_FROM_K": "ACTION",
    "MEMORY_OPS_RESUME": "ACTION",
    "MERGED": "OUTCOME",
    "INCIDENT_CLOSE": "OUTCOME",
    "blocked_needs_human": "OUTCOME",
    "INCIDENT_RESUME_FAIL": "OUTCOME",
    "STRATEGY_MISS": "SKIP",
    "LOW_ACC_STRATEGY_MISS": "SKIP",
}


def is_activation_marker(marker: str) -> bool:
    """True for wake/classify markers and follow-on babysitter actions."""
    return marker not in _ACTIVATIONS_SKIP


def _activation_role(marker: str) -> str:
    if marker in _ACTIVATION_ROLES:
        return _ACTIVATION_ROLES[marker]
    if marker.startswith("HELPER_FIRST_USE_K="):
        return "ACTION"
    return "ACTION"


class HumanLogWriter:
    """Append-only master + gsm/spider/smiles lane logs + activations.log."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.master_path = root / "logs" / "zero_acc_babysitter" / "master.log"
        self.activations_path = (
            root / "logs" / "zero_acc_babysitter" / "activations.log"
        )
        self.lane_paths = {
            "gsm": root / "logs" / "zero_acc_babysitter" / "lanes" / "gsm.log",
            "spider": root / "logs" / "zero_acc_babysitter" / "lanes" / "spider.log",
            "smiles": root / "logs" / "zero_acc_babysitter" / "lanes" / "smiles.log",
        }
        self.master_path.parent.mkdir(parents=True, exist_ok=True)
        for path in self.lane_paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)

    def _stamp(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def emit(self, cell_id: str, marker: str, detail: str = "") -> None:
        stamp = self._stamp()
        line = f"{stamp} cell={cell_id} {marker}"
        if detail:
            line = f"{line} {detail}"
        line = line.rstrip() + "\n"
        with self.master_path.open("a", encoding="utf-8") as handle:
            handle.write(line)

        try:
            lane = lane_name(cell_id)
        except ValueError:
            lane = None

        if is_activation_marker(marker):
            role = _activation_role(marker)
            act = f"{stamp} {role} cell={cell_id}"
            if lane is not None:
                act = f"{act} lane={lane}"
            act = f"{act} {marker}"
            if detail:
                act = f"{act} {detail}"
            act = act.rstrip() + "\n"
            with self.activations_path.open("a", encoding="utf-8") as handle:
                handle.write(act)

        if lane is None:
            # System markers (e.g. cell_id=watcher) go to master (+ activations if any).
            return
        with self.lane_paths[lane].open("a", encoding="utf-8") as handle:
            handle.write(line)

    def read_master(self) -> str:
        if not self.master_path.is_file():
            return ""
        return self.master_path.read_text(encoding="utf-8")

    def read_activations(self) -> str:
        if not self.activations_path.is_file():
            return ""
        return self.activations_path.read_text(encoding="utf-8")

    def read_lane(self, lane: str) -> str:
        path = self.lane_paths[lane]
        if not path.is_file():
            return ""
        return path.read_text(encoding="utf-8")
