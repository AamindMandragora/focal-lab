#!/usr/bin/env python3
"""Monitor approved GSM cycles and schedule one warm 40-attempt extension."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time


BILLING_ACCOUNT = "887730490125"


@dataclass(frozen=True)
class CellSpec:
    name: str
    output_name: str
    extension_output_name: str
    gpu: int
    min_free_mib: int
    base_history: str | None = None


CELL_SPECS = {
    "gsm14b": CellSpec(
        name="gsm14b",
        output_name="synth_gsm14b_z3bar_retry_0708_infraretry_kvfix_0711",
        extension_output_name="warm80_gsm14b_0713",
        gpu=2,
        min_free_mib=30000,
        base_history=".context/http429_resume_seeds/gsm14b_history_before33.json",
    ),
    "gsm-qwen35-4b": CellSpec(
        name="gsm-qwen35-4b",
        output_name="post14b_rebar_gsm-qwen35-4b_0711",
        extension_output_name="warm80_gsm-qwen35-4b_0713",
        gpu=3,
        min_free_mib=18000,
    ),
    "gsm-qwen35-9b": CellSpec(
        name="gsm-qwen35-9b",
        output_name="post14b_rebar_gsm-qwen35-9b_0711",
        extension_output_name="warm80_gsm-qwen35-9b_0713",
        gpu=1,
        min_free_mib=25000,
        base_history=".context/http429_resume_seeds/gsm-qwen35-9b_history_before5.json",
    ),
}


@dataclass(frozen=True)
class CycleStatus:
    state: str
    report: Path | None = None
    detail: str = ""


@dataclass(frozen=True)
class PreparedExtension:
    seed_file: Path
    history_file: Path
    manifest_file: Path


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def newest_report(root: Path, spec: CellSpec, filename: str) -> Path | None:
    output_root = root / "outputs/generated" / spec.output_name
    reports = list(output_root.glob(f"**/results/{filename}"))
    return max(reports, key=lambda path: path.stat().st_mtime) if reports else None


def report_has_evaluated_attempt(report: Path, attempt_number: int) -> bool:
    payload = json.loads(report.read_text(encoding="utf-8"))
    for attempt in payload.get("attempts", []):
        evaluation = attempt.get("evaluation") or {}
        if (
            int(attempt.get("attempt_number", -1)) == attempt_number
            and evaluation.get("num_examples", 0) > 0
            and evaluation.get("accuracy") is not None
            and evaluation.get("syntax_rate") is not None
        ):
            return True
    return False


def inspect_cycle(
    root: Path,
    spec: CellSpec,
    active_output_names: set[str],
    failure_not_before_epoch: float = 0,
) -> CycleStatus:
    success = newest_report(root, spec, "success_report.json")
    if success:
        return CycleStatus("success", success, "success_report.json exists")
    if spec.output_name in active_output_names:
        return CycleStatus("running", detail="matching synthesis process is alive")
    failure = newest_report(root, spec, "failure_report.json")
    if failure and failure.stat().st_mtime < failure_not_before_epoch:
        return CycleStatus("incomplete", failure, "failure report predates policy start")
    if failure and report_has_evaluated_attempt(failure, 40):
        return CycleStatus("failed", failure, "attempt 40 evaluated without success")
    return CycleStatus("incomplete", failure, "attempt 40 is not complete")


def decide_actions(statuses: dict[str, str], state: dict) -> list[str]:
    extensions = state.get("extensions", {})
    actions: list[str] = []
    if statuses.get("gsm14b") == "failed" and "gsm14b" not in extensions:
        actions.append("gsm14b")
    q4 = statuses.get("gsm-qwen35-4b")
    q9 = statuses.get("gsm-qwen35-9b")
    pair_unrecorded = not any(
        cell in extensions for cell in ("gsm-qwen35-4b", "gsm-qwen35-9b")
    )
    if q4 == "failed" and q9 == "failed" and pair_unrecorded:
        actions.extend(("gsm-qwen35-4b", "gsm-qwen35-9b"))
    return actions


def normalized_history_record(attempt: dict) -> dict | None:
    evaluation = attempt.get("evaluation") or {}
    if not attempt.get("strategy_code") or evaluation.get("num_examples", 0) <= 0:
        return None
    if evaluation.get("accuracy") is None or evaluation.get("syntax_rate") is None:
        return None
    return {
        "attempt_number": int(attempt["attempt_number"]),
        "strategy_code": str(attempt["strategy_code"]),
        "accuracy": float(evaluation["accuracy"]),
        "syntax_rate": float(evaluation["syntax_rate"]),
        "contains_delimiters": bool(evaluation.get("contains_delimiters", True)),
        "num_examples": int(evaluation["num_examples"]),
        "num_correct": int(
            evaluation.get(
                "num_correct",
                round(float(evaluation["accuracy"]) * int(evaluation["num_examples"])),
            )
        ),
        "total_time_seconds": float(evaluation.get("total_time_seconds", 0.0)),
        "timestamp": str(attempt.get("timestamp", "restored")),
    }


def prepare_extension(
    root: Path, spec: CellSpec, failure_report: Path
) -> PreparedExtension:
    payload = json.loads(failure_report.read_text(encoding="utf-8"))
    records: dict[int, dict] = {}
    if spec.base_history:
        base_path = root / spec.base_history
        if not base_path.exists():
            raise FileNotFoundError(f"missing base history for {spec.name}: {base_path}")
        for record in json.loads(base_path.read_text(encoding="utf-8")):
            records[int(record["attempt_number"])] = record

    attempt_40: dict | None = None
    for attempt in payload.get("attempts", []):
        number = int(attempt.get("attempt_number", -1))
        if number == 40 and normalized_history_record(attempt):
            attempt_40 = attempt
        elif number < 40:
            record = normalized_history_record(attempt)
            if record:
                records[number] = record
    if attempt_40 is None:
        raise ValueError(f"{failure_report} has no evaluated attempt 40 strategy")

    work = root / ".context/gsm_warm_extension_policy" / spec.name
    work.mkdir(parents=True, exist_ok=True)
    history_file = work / "history_before_attempt40.json"
    seed_file = work / "attempt40.dfy"
    manifest_file = work / "manifest.json"
    history_file.write_text(
        json.dumps([records[key] for key in sorted(records)], indent=2) + "\n",
        encoding="utf-8",
    )
    seed_file.write_text(str(attempt_40["strategy_code"]).rstrip() + "\n", encoding="utf-8")
    manifest = {
        "cell": spec.name,
        "source_failure_report": str(failure_report),
        "replay_attempt": 40,
        "first_new_attempt": 41,
        "last_new_attempt": 80,
        "new_iterations": 40,
        "cli_max_iterations": 41,
        "initial_attempt_offset": 39,
        "history_attempts": sorted(records),
        "billing_account": BILLING_ACCOUNT,
        "region": "us-east-1",
        "warm_override_scope": spec.name,
    }
    manifest_file.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return PreparedExtension(seed_file, history_file, manifest_file)


def load_state(path: Path) -> dict:
    if not path.exists():
        return {"extensions": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def active_output_names() -> set[str]:
    result = subprocess.run(
        ["ps", "-eo", "args="], text=True, capture_output=True, check=True
    )
    names = set()
    for line in result.stdout.splitlines():
        words = line.split()
        if "synthesis.run_synthesis" not in line or "--output-name" not in words:
            continue
        index = words.index("--output-name")
        if index + 1 < len(words):
            names.add(words[index + 1])
    return names


def gpu_free_mib(gpu: int) -> int | None:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu}",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        return None
    try:
        return int(result.stdout.strip())
    except ValueError:
        return None


def log(log_path: Path, message: str) -> None:
    line = f"{utc_now()} [gsm-warm-policy] {message}"
    print(line, flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def schedule_actions(
    root: Path,
    state_path: Path,
    state: dict,
    actions: list[str],
    statuses: dict[str, CycleStatus],
    log_path: Path,
) -> None:
    prepared = {}
    for cell in actions:
        report = statuses[cell].report
        if report is None:
            raise ValueError(f"missing failure report for scheduled cell {cell}")
        prepared[cell] = prepare_extension(root, CELL_SPECS[cell], report)
    for cell in actions:
        item = prepared[cell]
        state.setdefault("extensions", {})[cell] = {
            "status": "scheduled",
            "scheduled_at": utc_now(),
            "seed_file": str(item.seed_file),
            "history_file": str(item.history_file),
            "manifest_file": str(item.manifest_file),
        }
    save_state(state_path, state)
    log(log_path, f"SCHEDULED cells={','.join(actions)} state={state_path}")


def launch_scheduled(
    root: Path,
    state_path: Path,
    state: dict,
    log_path: Path,
    dry_run: bool,
) -> None:
    worker = root / "scripts/runtime/run_gsm_warm_extension.sh"
    for cell, record in state.get("extensions", {}).items():
        if record.get("status") != "scheduled":
            continue
        spec = CELL_SPECS[cell]
        free = gpu_free_mib(spec.gpu) if not dry_run else spec.min_free_mib
        if free is None or free < spec.min_free_mib:
            log(
                log_path,
                f"GPU_WAIT cell={cell} gpu={spec.gpu} free_mib={free} required={spec.min_free_mib}",
            )
            continue
        env = os.environ.copy()
        env.update(
            {
                "CELL": cell,
                "SEED_FILE": record["seed_file"],
                "HISTORY_FILE": record["history_file"],
                "OUTPUT_NAME": spec.extension_output_name,
                "GPU": str(spec.gpu),
                "CONFIRM_BEDROCK_ACCOUNT_887730490125": "yes",
                "REPO": str(root),
                "DRY_RUN": "1" if dry_run else "0",
            }
        )
        if dry_run:
            result = subprocess.run(
                ["bash", str(worker)], env=env, text=True, capture_output=True, check=True
            )
            record["status"] = "dry_run_ready"
            record["dry_run"] = json.loads(result.stdout)
            log(log_path, f"DRY_RUN_READY cell={cell} gpu={spec.gpu}")
        else:
            queue_log = root / "logs/gsm_warm_extension_queue.log"
            queue_handle = queue_log.open("a", encoding="utf-8")
            process = subprocess.Popen(
                ["bash", str(worker)],
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=queue_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            queue_handle.close()
            record["status"] = "launched"
            record["launched_at"] = utc_now()
            record["pid"] = process.pid
            log(log_path, f"LAUNCHED cell={cell} pid={process.pid} gpu={spec.gpu}")
        save_state(state_path, state)


def run_once(root: Path, state_path: Path, log_path: Path, dry_run: bool) -> None:
    state = load_state(state_path)
    if "policy_started_at_epoch" not in state:
        state["policy_started_at_epoch"] = time.time()
        state["policy_started_at"] = utc_now()
        save_state(state_path, state)
    active = active_output_names()
    statuses = {
        cell: inspect_cycle(
            root,
            spec,
            active,
            failure_not_before_epoch=float(state["policy_started_at_epoch"]),
        )
        for cell, spec in CELL_SPECS.items()
    }
    log(
        log_path,
        "STATUS " + " ".join(f"{cell}={status.state}" for cell, status in statuses.items()),
    )
    actions = decide_actions(
        {cell: status.state for cell, status in statuses.items()}, state
    )
    if actions:
        schedule_actions(root, state_path, state, actions, statuses, log_path)
    launch_scheduled(root, state_path, state, log_path, dry_run)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/home/aadivyar/csd-generation"))
    parser.add_argument("--state", type=Path)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = args.repo.resolve()
    state_path = args.state or root / ".context/gsm_warm_extension_policy_state.json"
    log_path = args.log or root / "logs/gsm_warm_extension_policy.log"
    log(
        log_path,
        f"START repo={root} account={BILLING_ACCOUNT} region=us-east-1 dry_run={args.dry_run}",
    )
    while True:
        try:
            run_once(root, state_path, log_path, args.dry_run)
        except Exception as exc:
            log(log_path, f"ERROR type={type(exc).__name__} detail={exc}")
            if args.once:
                raise
        if args.once:
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
