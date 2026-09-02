#!/usr/bin/env python3
"""Free GPU memory held by abandoned vLLM EngineCore processes."""

from __future__ import annotations

import argparse
import fcntl
import logging
import os
from dataclasses import dataclass
import signal
import subprocess
import time
from collections.abc import Callable, Iterable


LOG = logging.getLogger("vllm-orphan-reaper")


@dataclass(frozen=True)
class Process:
    pid: int
    ppid: int
    elapsed_seconds: int
    uid: int
    started_at: str
    command: str


def read_processes(pid: int | None = None) -> list[Process]:
    selection = ["-p", str(pid)] if pid is not None else ["-e"]
    output = subprocess.run(
        [
            "ps",
            *selection,
            "-o",
            "pid=,ppid=,etimes=,uid=,lstart=,args=",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    processes = []
    for line in output.splitlines():
        fields = line.strip().split(maxsplit=9)
        if len(fields) == 10:
            processes.append(
                Process(
                    *(int(value) for value in fields[:4]),
                    started_at=" ".join(fields[4:9]),
                    command=fields[9],
                )
            )
    return processes


def orphan_engines(
    processes: Iterable[Process], *, uid: int, grace_seconds: int
) -> list[Process]:
    return [
        process
        for process in processes
        if process.uid == uid
        and process.ppid == 1
        and process.elapsed_seconds >= grace_seconds
        and process.command == "VLLM::EngineCore"
    ]


def read_process(pid: int) -> Process | None:
    processes = read_processes(pid)
    return processes[0] if processes else None


def same_process(expected: Process, current: Process | None) -> bool:
    return current is not None and (
        current.pid,
        current.ppid,
        current.uid,
        current.started_at,
        current.command,
    ) == (
        expected.pid,
        expected.ppid,
        expected.uid,
        expected.started_at,
        expected.command,
    )


def reap_process(
    process: Process,
    *,
    send_signal: Callable[[int, int], None] = os.kill,
    read_process: Callable[[int], Process | None] = read_process,
    wait: Callable[[float], None] = time.sleep,
    term_wait_seconds: float = 5,
) -> None:
    pid = process.pid
    if not same_process(process, read_process(pid)):
        return
    LOG.warning("[vllm-orphan-reaper] terminating orphan pid=%d", pid)
    try:
        send_signal(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    wait(term_wait_seconds)
    if same_process(process, read_process(pid)):
        LOG.warning("[vllm-orphan-reaper] force-killing orphan pid=%d", pid)
        try:
            send_signal(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def run_forever(*, poll_seconds: float, grace_seconds: int) -> None:
    uid = os.getuid()
    while True:
        try:
            candidates = orphan_engines(
                read_processes(), uid=uid, grace_seconds=grace_seconds
            )
            LOG.info(
                "[vllm-orphan-reaper] scan uid=%d candidates=%s",
                uid,
                [process.pid for process in candidates],
            )
            for process in candidates:
                reap_process(process)
        except Exception:
            LOG.exception("[vllm-orphan-reaper] scan failed")
        time.sleep(poll_seconds)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=float, default=5)
    parser.add_argument("--grace-seconds", type=int, default=30)
    parser.add_argument("--lock-file", default="/tmp/csd-vllm-orphan-reaper.lock")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    lock = open(args.lock_file, "w", encoding="utf-8")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        LOG.info("[vllm-orphan-reaper] another monitor already owns the lock")
        return 0
    lock.write(str(os.getpid()))
    lock.flush()
    run_forever(poll_seconds=args.poll_seconds, grace_seconds=args.grace_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
