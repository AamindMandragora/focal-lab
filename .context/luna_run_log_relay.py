#!/usr/bin/env python3
"""Relay only the current and future portion of each focal synthesis run log."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path


ATTEMPT = re.compile(rb"(?m)^[ \t]*Attempt \d+/\d+[ \t]*$")
CONTEXT_BYTES = 512 * 1024
ACTIVE_WITHIN_SECONDS = 3 * 24 * 60 * 60
SEARCH_CHUNK_BYTES = 256 * 1024


def latest_attempt_line(source: Path, size: int) -> bytes:
    with source.open("rb") as handle:
        end = size
        overlap = b""
        while end > 0:
            start = max(0, end - SEARCH_CHUNK_BYTES)
            handle.seek(start)
            chunk = handle.read(end - start) + overlap
            matches = list(ATTEMPT.finditer(chunk))
            if matches:
                return matches[-1].group(0).strip()
            overlap = chunk[:64]
            end = start
    return b""


def cursor_fingerprint(source: Path, position: int) -> str:
    start = max(0, position - 4096)
    with source.open("rb") as handle:
        handle.seek(start)
        return hashlib.sha256(handle.read(position - start)).hexdigest()


class RunLogRelay:
    def __init__(self, generated_dir: Path, relay_dir: Path, state_path: Path) -> None:
        self.generated_dir = generated_dir
        self.relay_dir = relay_dir
        self.state_path = state_path
        try:
            self.state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self.state = {"sources": {}}

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.state, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(self.state_path)

    def poll_once(self) -> int:
        copied = 0
        self.relay_dir.mkdir(parents=True, exist_ok=True)
        for source in sorted(self.generated_dir.glob("*/run.log")):
            stat = source.stat()
            if time.time() - stat.st_mtime > ACTIVE_WITHIN_SECONDS:
                continue
            run_name = source.parent.name
            destination = self.relay_dir / f"{run_name}.log"
            identity = [stat.st_dev, stat.st_ino]
            with source.open("rb") as handle:
                head_hash = hashlib.sha256(handle.read(1024)).hexdigest()
            saved = self.state["sources"].get(run_name)
            saved_position = int(saved.get("position", 0)) if saved else 0
            if saved is None or saved.get("identity") != identity or stat.st_size < saved_position or saved.get("head_hash") not in (None, head_hash) or saved.get("cursor_hash") not in (None, cursor_fingerprint(source, min(saved_position, stat.st_size))):
                with source.open("rb") as handle:
                    start = max(0, stat.st_size - CONTEXT_BYTES)
                    handle.seek(start)
                    context = handle.read()
                context_matches = list(ATTEMPT.finditer(context))
                if context_matches:
                    context = context[context_matches[-1].start():]
                    prefix = b""
                else:
                    attempt_line = latest_attempt_line(source, stat.st_size)
                    prefix = attempt_line + b"\n" if attempt_line else b""
                destination.write_bytes(prefix + context)
                position = stat.st_size
            else:
                position = int(saved["position"])
            if position < stat.st_size:
                with source.open("rb") as handle:
                    handle.seek(position)
                    update = handle.read()
                    position = handle.tell()
                with destination.open("ab") as handle:
                    handle.write(update)
                copied += len(update)
            self.state["sources"][run_name] = {"identity": identity, "position": position, "head_hash": head_hash, "cursor_hash": cursor_fingerprint(source, position)}
        self._save()
        return copied


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--relay-dir", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=5)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    relay = RunLogRelay(args.generated_dir, args.relay_dir, args.state)
    while True:
        relay.poll_once()
        if args.once:
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
