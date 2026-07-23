"""Human-readable master + per-lane logs with babysitter event markers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from scripts.runtime.zero_acc_babysitter.domains import lane_name


class HumanLogWriter:
    """Append-only master + gsm/spider/smiles lane logs."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.master_path = root / "logs" / "zero_acc_babysitter" / "master.log"
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
        line = f"{self._stamp()} cell={cell_id} {marker}"
        if detail:
            line = f"{line} {detail}"
        line = line.rstrip() + "\n"
        with self.master_path.open("a", encoding="utf-8") as handle:
            handle.write(line)
        try:
            lane = lane_name(cell_id)
        except ValueError:
            # System markers (e.g. cell_id=watcher) go to master only.
            return
        with self.lane_paths[lane].open("a", encoding="utf-8") as handle:
            handle.write(line)

    def read_master(self) -> str:
        if not self.master_path.is_file():
            return ""
        return self.master_path.read_text(encoding="utf-8")

    def read_lane(self, lane: str) -> str:
        path = self.lane_paths[lane]
        if not path.is_file():
            return ""
        return path.read_text(encoding="utf-8")
