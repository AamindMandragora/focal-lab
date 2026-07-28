#!/usr/bin/env python3
"""Drain the GPU3 retry queue written by run_all_tests.py."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import shutil


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_QUEUE = ROOT_DIR / "outputs" / "gpu3_retry_queue.jsonl"


def load_json(path: Path) -> dict:
    if not path.is_file():
        return {"attempted_ids": []}
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return {"attempted_ids": []}
    payload.setdefault("attempted_ids", [])
    return payload


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_queue(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    records = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            print(f"[warn] ignoring malformed queue line {line_number}: {exc}", file=sys.stderr)
            continue
        if isinstance(record.get("cmd"), list) and record.get("id"):
            records.append(record)
    return records


def gpu_used_mb(gpu: str) -> int | None:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None
    result = subprocess.run(
        [
            nvidia_smi,
            f"--id={gpu}",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    first_line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    try:
        return int(first_line.strip())
    except ValueError:
        return None


def wait_for_gpu(args: argparse.Namespace) -> None:
    while True:
        used_mb = gpu_used_mb(str(args.gpu))
        if used_mb is None or used_mb <= args.max_used_mb:
            return
        print(
            f"[gpu3-retry] GPU {args.gpu} is busy ({used_mb} MiB used); "
            f"waiting {args.wait_seconds}s"
        )
        time.sleep(args.wait_seconds)


PRECHECK_TIMEOUT_SECONDS = 180


def precheck_cmd(cmd: list[str]) -> str | None:
    """Catch what is knowable in seconds, before the GPU wait and the model load.

    Two things used to surface only after waiting for the GPU to free up and
    then paying a multi-minute model load:

      1. a flag the launcher does not have -- it dies at argparse
      2. a module that does not exist

    The second is the worse one. A missing module has been swallowed into a
    fake 0% score in this project before, which reads as a model result and
    costs weeks rather than minutes.

    Returns None when the command looks launchable, otherwise the reason it
    does not. Deliberately loose: a check that wrongly blocks a good run is
    worse than one that lets a bad run through, since the run itself is still
    the real test.
    """
    module = None
    for i, part in enumerate(cmd):
        if part == "-m" and i + 1 < len(cmd):
            module = cmd[i + 1]
            break
    if module is None:
        return None  # not a `python -m ...` launch; nothing cheap to check here

    python = cmd[0]

    def _probe(argv: list[str]) -> subprocess.CompletedProcess | None:
        try:
            return subprocess.run(
                argv,
                cwd=ROOT_DIR,
                capture_output=True,
                text=True,
                timeout=PRECHECK_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            return None

    # 1. Does it import at all? This is the whole missing-module bug class.
    imported = _probe([python, "-c", f"import {module}"])
    if imported is None:
        return f"importing {module} timed out after {PRECHECK_TIMEOUT_SECONDS}s"
    if imported.returncode != 0:
        last = (imported.stderr.strip().splitlines() or ["no output"])[-1]
        return f"{module} does not import: {last}"

    # 2. Does it accept every flag being passed? --help runs argparse and
    #    exits, so this costs nothing next to loading a model.
    helped = _probe([python, "-m", module, "--help"])
    if helped is None:
        return f"{module} --help timed out after {PRECHECK_TIMEOUT_SECONDS}s"
    if helped.returncode != 0:
        last = (helped.stderr.strip().splitlines() or ["no output"])[-1]
        return f"{module} --help failed, so the run cannot start: {last}"

    help_text = helped.stdout + helped.stderr
    unknown = [
        part for part in cmd
        if part.startswith("--") and part.split("=")[0] not in help_text
    ]
    if unknown:
        return f"{module} has no such flag(s): {', '.join(unknown)}"

    return None


def run_record(record: dict, *, gpu: str, log_dir: Path) -> int:
    cmd = [str(part) for part in record["cmd"]]
    record_id = str(record["id"])
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"gpu3_retry_{record_id}_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["VAS_MAX_CUDA_DEVICES"] = "1"
    env.setdefault("PYTHONUNBUFFERED", "1")
    print(f"[gpu3-retry] running {record_id}: {' '.join(cmd)}")
    print(f"[gpu3-retry] log: {log_path}")
    with log_path.open("w") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT_DIR,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log_file.write(line)
        return proc.wait()


def drain_once(args: argparse.Namespace) -> int:
    state_path = Path(args.state)
    state = load_json(state_path)
    attempted = set(state.get("attempted_ids", []))
    for record in read_queue(Path(args.queue)):
        record_id = str(record["id"])
        if record_id in attempted:
            continue

        # Before the GPU wait, not after it. Waiting for a free GPU and then
        # loading a model only to die at argparse is the expensive way to find
        # out something a --help call answers in seconds.
        refusal = precheck_cmd([str(part) for part in record["cmd"]])
        if refusal is not None:
            print(f"[gpu3-retry] SKIPPING {record_id}: {refusal}")
            attempted.add(record_id)
            state["attempted_ids"] = sorted(attempted)
            state.setdefault("attempts", {})[record_id] = {
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "return_code": None,
                "precheck_refusal": refusal,
                "reason": record.get("reason"),
                "case": record.get("case", {}),
            }
            save_json(state_path, state)
            continue

        wait_for_gpu(args)
        started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return_code = run_record(record, gpu=str(args.gpu), log_dir=Path(args.log_dir))
        attempted.add(record_id)
        state["attempted_ids"] = sorted(attempted)
        state.setdefault("attempts", {})[record_id] = {
            "started_at": started_at,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "return_code": return_code,
            "reason": record.get("reason"),
            "case": record.get("case", {}),
        }
        save_json(state_path, state)
        return 0 if return_code == 0 else return_code
    return 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run queued failed MetaDecode cells on GPU3.")
    parser.add_argument("--queue", default=os.environ.get("CSD_GPU3_RETRY_QUEUE", str(DEFAULT_QUEUE)))
    parser.add_argument("--state", default=os.environ.get("CSD_GPU3_RETRY_STATE", str(DEFAULT_QUEUE) + ".state.json"))
    parser.add_argument("--log-dir", default=os.environ.get("CSD_GPU3_RETRY_LOG_DIR", str(ROOT_DIR / "logs" / "gpu3_retry_queue")))
    parser.add_argument("--gpu", default=os.environ.get("CSD_GPU3_RETRY_DEVICE", "3"))
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--wait-seconds", type=int, default=60)
    parser.add_argument("--max-used-mb", type=int, default=int(os.environ.get("CSD_GPU3_RETRY_MAX_USED_MB", "1024")))
    parser.add_argument("--once", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    while True:
        code = drain_once(args)
        if args.once:
            return code
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
