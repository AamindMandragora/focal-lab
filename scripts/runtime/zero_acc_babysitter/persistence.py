"""Incident + per-cell JSON persistence for mid-crash resume."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from scripts.runtime.zero_acc_babysitter.constants import (
    CellState,
    IncidentPhase,
    PathKind,
)


@dataclass
class IncidentRecord:
    incident_id: str
    cell_id: str
    attempt_index: int
    path_kind: str
    trigger_unix_ts: int
    state: str = IncidentPhase.OPEN.value
    phase: str = IncidentPhase.OPEN.value
    cloud_attempt_count: int = 0
    pr_url: str | None = None
    broken_sha: str | None = None
    helper_first_use_k: int | None = None
    closed: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "IncidentRecord":
        known = {
            "incident_id",
            "cell_id",
            "attempt_index",
            "path_kind",
            "trigger_unix_ts",
            "state",
            "phase",
            "cloud_attempt_count",
            "pr_url",
            "broken_sha",
            "helper_first_use_k",
            "closed",
            "extra",
        }
        kwargs = {k: payload.get(k) for k in known if k in payload}
        kwargs.setdefault("extra", {k: v for k, v in payload.items() if k not in known})
        return cls(**kwargs)  # type: ignore[arg-type]


@dataclass
class CellRecord:
    cell_id: str
    state: str = CellState.PENDING.value
    active_incident_id: str | None = None
    kill_count: int = 0
    restart_k: int | None = None
    scored_under_sha: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CellRecord":
        return cls(
            cell_id=str(payload["cell_id"]),
            state=str(payload.get("state", CellState.PENDING.value)),
            active_incident_id=payload.get("active_incident_id"),
            kill_count=int(payload.get("kill_count", 0)),
            restart_k=payload.get("restart_k"),
            scored_under_sha=payload.get("scored_under_sha"),
        )


class IncidentStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.incidents_dir = root / "logs" / "zero_acc_babysitter" / "incidents"
        self.cells_dir = root / "logs" / "zero_acc_babysitter" / "cells"
        self.incidents_dir.mkdir(parents=True, exist_ok=True)
        self.cells_dir.mkdir(parents=True, exist_ok=True)

    def incident_path(self, incident_id: str) -> Path:
        safe = incident_id.replace("/", "_")
        return self.incidents_dir / f"{safe}.json"

    def cell_path(self, cell_id: str) -> Path:
        return self.cells_dir / f"{cell_id}.json"

    def save_incident(self, record: IncidentRecord) -> None:
        path = self.incident_path(record.incident_id)
        path.write_text(json.dumps(record.to_dict(), indent=2) + "\n", encoding="utf-8")

    def load_incident(self, incident_id: str) -> IncidentRecord | None:
        path = self.incident_path(incident_id)
        if not path.is_file():
            return None
        return IncidentRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def save_cell(self, record: CellRecord) -> None:
        path = self.cell_path(record.cell_id)
        path.write_text(json.dumps(record.to_dict(), indent=2) + "\n", encoding="utf-8")

    def load_cell(self, cell_id: str) -> CellRecord | None:
        path = self.cell_path(cell_id)
        if not path.is_file():
            return None
        return CellRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def list_open_incidents(self) -> list[IncidentRecord]:
        out: list[IncidentRecord] = []
        for path in self.incidents_dir.glob("*.json"):
            record = IncidentRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))
            if not record.closed:
                out.append(record)
        return out


def make_incident_id(
    cell_id: str, attempt_index: int, path_kind: PathKind, trigger_unix_ts: int | None = None
) -> str:
    ts = int(time.time()) if trigger_unix_ts is None else int(trigger_unix_ts)
    return f"{cell_id}:{attempt_index}:{path_kind.value}:{ts}"
