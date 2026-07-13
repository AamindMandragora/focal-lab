#!/usr/bin/env python3
"""Live HTML dashboard for CSD experiment runs.

The dashboard intentionally uses only the Python standard library so it can run
on focal without installing frontend or web dependencies.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import subprocess
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


ROOT_DIR = Path(__file__).resolve().parents[1]
QUEUE_PATH = ROOT_DIR / "outputs" / "gpu3_retry_queue.jsonl"
STATE_PATH = ROOT_DIR / "outputs" / "gpu3_retry_queue.jsonl.state.json"
LOG_DIR = ROOT_DIR / "logs"
RESEARCH_TRACKER_PATH = ROOT_DIR / "saved-results" / "research_tracker_status.json"
REPORT_GLOBS = ("success_report.json", "failure_report.json")
RUN_PATTERNS = (
    "run_all_tests.py",
    "synthesis.run_synthesis",
    "run_legacy_fixed_strategy",
    "run_gpu3_retry_queue.py",
    "experiment_dashboard.py",
)


def safe_read_text(path: Path, *, limit_bytes: int | None = None) -> str:
    try:
        if limit_bytes is None:
            return path.read_text(errors="replace")
        size = path.stat().st_size
        with path.open("rb") as handle:
            if size > limit_bytes:
                handle.seek(size - limit_bytes)
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def load_json(path: Path) -> object | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    for line_number, line in enumerate(safe_read_text(path).splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            records.append({"line_number": line_number, "malformed": True, "error": str(exc)})
            continue
        if isinstance(payload, dict):
            payload.setdefault("line_number", line_number)
            records.append(payload)
    return records


def run_command(args: list[str]) -> str:
    try:
        result = subprocess.run(args, text=True, capture_output=True, check=False, timeout=8)
    except Exception as exc:
        return f"ERROR: {exc}"
    if result.returncode != 0 and not result.stdout:
        return result.stderr.strip()
    return result.stdout.strip()


def parse_gpu_rows() -> list[dict]:
    output = run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    rows: list[dict] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 6:
            continue
        index, uuid, name, used, total, util = parts
        rows.append(
            {
                "index": index,
                "uuid": uuid,
                "name": name,
                "memory_used_mb": int(used) if used.isdigit() else used,
                "memory_total_mb": int(total) if total.isdigit() else total,
                "utilization_pct": int(util) if util.isdigit() else util,
            }
        )
    return rows


def parse_gpu_processes() -> list[dict]:
    output = run_command(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    rows: list[dict] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        uuid, pid, name, mem = parts
        rows.append({"gpu_uuid": uuid, "pid": pid, "process_name": name, "memory_mb": mem})
    return rows


def parse_etime_seconds(etime: str) -> int:
    """Convert a `ps` ELAPSED field ([[DD-]hh:]mm:ss) into seconds.

    Cross-platform: Linux `etimes` gives integer seconds directly, but macOS `ps`
    has no `etimes` keyword, so we read the formatted `etime` field instead.
    """
    etime = etime.strip()
    days = 0
    if "-" in etime:
        day_part, etime = etime.split("-", 1)
        days = int(day_part)
    parts = [int(part) for part in etime.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    hours, minutes, seconds = parts[-3:]
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def parse_process_table() -> list[dict]:
    output = run_command(["ps", "-eo", "pid,ppid,etime,command"])
    rows: list[dict] = []
    for line in output.splitlines()[1:]:
        match = re.match(r"\s*(\d+)\s+(\d+)\s+(\S+)\s+(.*)", line)
        if not match:
            continue
        pid, ppid, etime, cmd = match.groups()
        rows.append({"pid": pid, "ppid": ppid, "elapsed_seconds": parse_etime_seconds(etime), "cmd": cmd})
    return rows


def parse_processes() -> list[dict]:
    rows: list[dict] = []
    for process in parse_process_table():
        if any(pattern in process["cmd"] for pattern in RUN_PATTERNS):
            rows.append(process)
    return rows


def extract_flag(cmd: str, flag: str) -> str | None:
    pattern = rf"(?:^|\s){re.escape(flag)}(?:=|\s+)(\"[^\"]+\"|'[^']+'|\S+)"
    match = re.search(pattern, cmd)
    if not match:
        return None
    return match.group(1).strip("'\"")


def has_flag(cmd: str, flag: str) -> bool:
    return re.search(rf"(?:^|\s){re.escape(flag)}(?:\s|$)", cmd) is not None


def classify_run(process: dict) -> dict:
    cmd = process.get("cmd", "")
    generated_output_dir = extract_flag(cmd, "--generated-output-dir")
    output_name = extract_flag(cmd, "--output-name")
    return {
        **process,
        "dataset": extract_flag(cmd, "--dataset") or extract_flag(cmd, "--benchmarks"),
        "model": extract_flag(cmd, "--eval-model") or extract_flag(cmd, "--models"),
        "generation_model": extract_flag(cmd, "--generation-model") or extract_flag(cmd, "--generation-models"),
        "max_iterations": extract_flag(cmd, "--max-iterations") or extract_flag(cmd, "--main-synthesis-iterations"),
        "max_steps": extract_flag(cmd, "--eval-max-steps") or extract_flag(cmd, "--eval-max-steps-gsm"),
        "eval_sample_size": extract_flag(cmd, "--eval-sample-size"),
        "eval_step_token_budget": extract_flag(cmd, "--eval-step-token-budget") or extract_flag(cmd, "--token-budgets"),
        "accuracy_win_margin": extract_flag(cmd, "--accuracy-win-margin"),
        "ablation_sections": extract_flag(cmd, "--ablation-sections"),
        "strategies": extract_flag(cmd, "--strategies"),
        "synthesis_iterations": extract_flag(cmd, "--synthesis-iterations"),
        "generated_output_dir": generated_output_dir,
        "ablation_output_dir": extract_flag(cmd, "--ablation-output-dir"),
        "skip_main": has_flag(cmd, "--skip-main"),
        "skip_ablations": has_flag(cmd, "--skip-ablations") or has_flag(cmd, "--skip-ablation"),
        "output": output_name or generated_output_dir,
    }


def process_title(process: dict | None) -> str:
    if not process:
        return "Idle"
    dataset = process.get("dataset") or "unknown"
    model = process.get("model") or ""
    gen = process.get("generation_model") or ""
    pieces: list[str] = []
    if "run_all_tests.py" in (process.get("cmd") or ""):
        if process.get("ablation_sections"):
            pieces.append(f"ablation {process['ablation_sections']}")
        elif process.get("skip_ablations"):
            pieces.append("main matrix")
        else:
            pieces.append("matrix")
    pieces.append(str(dataset))
    if model:
        pieces.append(f"model={str(model).split('/')[-1]}")
    elif "run_all_tests.py" in (process.get("cmd") or ""):
        pieces.append("model=default")
    if gen:
        pieces.append(f"gen={gen}")
    if process.get("strategies"):
        pieces.append(f"strategy={process['strategies']}")
    if process.get("max_iterations"):
        pieces.append(f"iter={process['max_iterations']}")
    if process.get("max_steps"):
        pieces.append(f"steps={process['max_steps']}")
    if process.get("accuracy_win_margin"):
        pieces.append(f"margin={process['accuracy_win_margin']}")
    return " / ".join(pieces)


METRIC_RE = re.compile(
    r"Attempt\s+(?P<attempt>\d+)/(?:\d+)|"
    r"Accuracy:\s+(?P<accuracy>[0-9.]+)%|"
    r"Syntax:\s+(?P<syntax>[0-9.]+)%|"
    r"SYNTHESIS\s+(?P<status>FAILED|SUCCEEDED)|"
    r"Full report saved to:\s+(?P<report>\S+)"
)

ALERT_PATTERNS = (
    ("api_retry", re.compile(r"\[api-retry\]|HTTP\s+(?:429|500|502|503|504|529)", re.IGNORECASE)),
    ("credits", re.compile(r"RESOURCE_EXHAUSTED|credits? (?:are )?depleted|quota|billing|prepayment", re.IGNORECASE)),
    ("traceback", re.compile(r"Traceback \(most recent call last\)|\bException\b|\bRuntimeError\b", re.IGNORECASE)),
    ("cuda_oom", re.compile(r"CUDA out of memory|out of memory|CUBLAS_STATUS_ALLOC_FAILED|NCCL error", re.IGNORECASE)),
    ("process_failed", re.compile(r"Exit code:\s*[1-9]\d*|returned non-zero exit status|Command failed", re.IGNORECASE)),
    ("threshold_failed", re.compile(r"SYNTHESIS FAILED|Evaluation below threshold|target accuracy unreachable", re.IGNORECASE)),
)


def runtime_alerts_from_lines(lines: list[str], *, max_alerts: int = 8) -> list[dict]:
    alerts: list[dict] = []
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        for kind, pattern in ALERT_PATTERNS:
            if pattern.search(stripped):
                alerts.append(
                    {
                        "kind": kind,
                        "line": line_number,
                        "message": stripped[-500:],
                    }
                )
                break
    return alerts[-max_alerts:]


def summarize_log(path: Path) -> dict:
    text = safe_read_text(path, limit_bytes=1_500_000)
    lines = text.splitlines()
    summary: dict = {
        "path": str(path.relative_to(ROOT_DIR)),
        "mtime": path.stat().st_mtime,
        "tail": lines[-80:],
        "attempt": None,
        "accuracy_pct": None,
        "syntax_pct": None,
        "status": None,
        "report": None,
        "recent_metrics": [],
        "alerts": runtime_alerts_from_lines(lines),
    }
    recent: list[str] = []
    for line in lines:
        if "Attempt " in line or "Accuracy:" in line or "Syntax:" in line or "SYNTHESIS " in line:
            recent.append(line.strip())
        for match in METRIC_RE.finditer(line):
            data = match.groupdict()
            if data.get("attempt"):
                summary["attempt"] = int(data["attempt"])
            if data.get("accuracy"):
                summary["accuracy_pct"] = float(data["accuracy"])
            if data.get("syntax"):
                summary["syntax_pct"] = float(data["syntax"])
            if data.get("status"):
                summary["status"] = data["status"].lower()
            if data.get("report"):
                summary["report"] = data["report"]
    summary["recent_metrics"] = recent[-30:]
    return summary


def attempt_stats_from_metrics(metrics: list[str]) -> dict:
    attempts: list[dict] = []
    current: dict | None = None
    for line in metrics:
        attempt_match = re.search(r"Attempt\s+(\d+)/(?:\d+)", line)
        if attempt_match:
            current = {"attempt": int(attempt_match.group(1)), "accuracy_pct": None, "syntax_pct": None}
            attempts.append(current)
            continue
        if current is None:
            continue
        acc_match = re.search(r"Accuracy:\s+([0-9.]+)%", line)
        syn_match = re.search(r"Syntax:\s+([0-9.]+)%", line)
        if acc_match:
            current["accuracy_pct"] = float(acc_match.group(1))
        if syn_match:
            current["syntax_pct"] = float(syn_match.group(1))
    complete = [item for item in attempts if item.get("accuracy_pct") is not None or item.get("syntax_pct") is not None]
    best_accuracy = max((item for item in complete if item.get("accuracy_pct") is not None), key=lambda item: item["accuracy_pct"], default=None)
    best_syntax_passing = max(
        (item for item in complete if item.get("syntax_pct") is not None and item["syntax_pct"] >= 90.0),
        key=lambda item: item.get("accuracy_pct") or -1,
        default=None,
    )
    latest = complete[-1] if complete else (attempts[-1] if attempts else None)
    return {
        "attempts": complete[-12:],
        "latest": latest,
        "best_accuracy": best_accuracy,
        "best_syntax_passing": best_syntax_passing,
    }


def rationale_snippets_from_prompt_io(path: Path) -> list[str]:
    snippets: list[str] = []
    for line in safe_read_text(path, limit_bytes=1_200_000).splitlines()[-120:]:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        output = record.get("output")
        if not isinstance(output, str) or not output.strip():
            continue
        text = output.strip()
        start = text.find("CSD_RATIONALE_BEGIN")
        if start >= 0:
            end = text.find("CSD_RATIONALE_END", start)
            text = text[start:end if end >= 0 else start + 900]
        elif not any(word in text.lower() for word in ("rationale", "attempt", "best", "regress", "revert", "failure")):
            continue
        text = re.sub(r"\s+", " ", text)
        snippets.append(text[:650])
    return snippets[-6:]


def rationale_trend(snippets: list[str], stats: dict) -> dict:
    joined = " ".join(snippets).lower()
    latest = stats.get("latest") or {}
    best = stats.get("best_accuracy") or {}
    label = "not enough rationale signal"
    tone = "warn"
    reasons: list[str] = []
    if any(word in joined for word in ("revert", "verbatim", "cycle", "cycling", "regressed", "deviation")):
        label = "cycling around prior best"
        tone = "warn"
        reasons.append("recent rationale mentions reverting/regression")
    if any(word in joined for word in ("best-known", "best prior", "dominant failure", "failure-mode", "single targeted", "diagnosis")):
        if label == "not enough rationale signal":
            label = "sensible local diagnosis"
            tone = "ok"
        reasons.append("references best attempts and failure modes")
    if latest.get("accuracy_pct") is not None and best.get("accuracy_pct") is not None:
        if latest["accuracy_pct"] + 5 < best["accuracy_pct"]:
            label = "regressing from best metric"
            tone = "bad"
            reasons.append("latest accuracy is materially below best")
        elif latest["accuracy_pct"] >= best["accuracy_pct"]:
            label = "metric trend improving"
            tone = "ok"
            reasons.append("latest matches or exceeds best accuracy")
    if not snippets:
        reasons.append("no recent rationale snippets found")
    return {"label": label, "tone": tone, "reasons": reasons[:3], "snippets": snippets}


def latest_logs() -> list[dict]:
    candidates: list[Path] = []
    candidates.extend(LOG_DIR.glob("*.log"))
    candidates.extend((LOG_DIR / "gpu3_retry_queue").glob("*.log"))
    for path in LOG_DIR.glob("*/prompt_io.jsonl"):
        candidates.append(path)
    existing = [path for path in candidates if path.is_file()]
    existing.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return [summarize_log(path) for path in existing[:10]]


def all_recent_logs(limit: int = 40) -> list[dict]:
    candidates: list[Path] = []
    candidates.extend(LOG_DIR.glob("*.log"))
    candidates.extend((LOG_DIR / "gpu3_retry_queue").glob("*.log"))
    candidates.extend(LOG_DIR.glob("*/prompt_io.jsonl"))
    existing = [path for path in candidates if path.is_file()]
    existing.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return [summarize_log(path) for path in existing[:limit]]


def prompt_io_for_process(process: dict) -> Path | None:
    output = process.get("output") or ""
    candidates = sorted(LOG_DIR.glob("*/prompt_io.jsonl"), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in candidates:
        if output and output in str(path):
            return path
    if "run_all_tests.py" in (process.get("cmd") or ""):
        return None
    dataset = process.get("dataset") or ""
    for path in candidates:
        text = str(path).replace("-", "_")
        if dataset and dataset.replace("-", "_") in text:
            return path
    return candidates[0] if candidates else None


def log_for_process(process: dict, logs: list[dict]) -> dict | None:
    output = process.get("output") or ""
    if output:
        for log in logs:
            if output in log["path"]:
                return log
    cmd = process.get("cmd", "")
    if "run_gpu3_retry_queue.py" in cmd:
        for log in logs:
            if "gpu3_retry_queue_worker.log" in log["path"]:
                return log
    if "run_all_tests.py" in cmd:
        generated_dir = extract_flag(cmd, "--generated-output-dir") or ""
        for log in logs:
            if generated_dir and generated_dir in "\n".join(log.get("tail", [])):
                return log
        for log in logs:
            if "gpu3_retry_queue" in log["path"]:
                return log
    return logs[0] if logs else None


def metric_value(evaluation: dict, key: str) -> float:
    value = evaluation.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return -1.0


def attempt_evaluations(payload: dict) -> list[tuple[int | None, dict]]:
    attempts = payload.get("attempts")
    if not isinstance(attempts, list):
        return []
    evaluated: list[tuple[int | None, dict]] = []
    for attempt in attempts:
        if not isinstance(attempt, dict):
            continue
        evaluation = attempt.get("evaluation")
        if isinstance(evaluation, dict):
            evaluated.append((attempt.get("attempt_number"), evaluation))
    return evaluated


def best_attempt_evaluation(payload: dict) -> tuple[int | None, dict | None]:
    evaluated = attempt_evaluations(payload)
    if not evaluated:
        return None, None
    return max(
        evaluated,
        key=lambda item: (
            metric_value(item[1], "accuracy"),
            metric_value(item[1], "syntax_rate"),
            metric_value(item[1], "num_correct"),
        ),
    )


def report_run_metadata(payload: dict, path: Path) -> dict:
    config = payload.get("run_configuration")
    if not isinstance(config, dict):
        config = {}
    thresholds = config.get("thresholds")
    if not isinstance(thresholds, dict):
        thresholds = {}
    author_model = config.get("author_model")
    if not isinstance(author_model, dict):
        author_model = {}
    evaluation_config = config.get("evaluation")
    if not isinstance(evaluation_config, dict):
        evaluation_config = {}
    synthesis_controls = config.get("synthesis_controls")
    if not isinstance(synthesis_controls, dict):
        synthesis_controls = {}

    output_name = config.get("output_name") or path.parent.parent.name
    path_text = str(path)
    dataset = evaluation_config.get("dataset")
    if not dataset:
        if "gsm_symbolic" in path_text:
            dataset = "gsm_symbolic"
        elif "spider" in path_text or "sql" in path_text:
            dataset = "spider"

    return {
        "output_name": output_name,
        "dataset": dataset,
        "eval_model": evaluation_config.get("eval_model"),
        "eval_backend": evaluation_config.get("eval_backend"),
        "eval_sample_size": evaluation_config.get("eval_sample_size"),
        "eval_max_steps": evaluation_config.get("eval_max_steps"),
        "eval_step_token_budget": evaluation_config.get("eval_step_token_budget"),
        "eval_max_seconds_per_example": evaluation_config.get("eval_max_seconds_per_example"),
        "min_examples_before_threshold_stop": evaluation_config.get("min_examples_before_threshold_stop"),
        "max_iterations": config.get("max_iterations"),
        "min_accuracy": thresholds.get("min_accuracy"),
        "min_syntax_rate": thresholds.get("min_syntax_rate"),
        "require_delimiters": thresholds.get("require_delimiters"),
        "author_backend": author_model.get("backend"),
        "author_model": author_model.get("model"),
        "author_thinking": author_model.get("anthropic_thinking"),
        "author_effort": author_model.get("anthropic_effort"),
        "helper_selection_policy": synthesis_controls.get("helper_selection_policy"),
        "adaptive_helper_mask": synthesis_controls.get("adaptive_helper_mask"),
        "refinement_beam_size": synthesis_controls.get("refinement_beam_size"),
        "restart_after_stuck_iters": synthesis_controls.get("restart_after_stuck_iters"),
        "restart_cooldown_iters": synthesis_controls.get("restart_cooldown_iters"),
    }


def report_summary(path: Path) -> dict:
    payload = load_json(path)
    result: dict = {
        "path": str(path.relative_to(ROOT_DIR)),
        "mtime": path.stat().st_mtime,
        "kind": path.name.replace("_report.json", ""),
    }
    if isinstance(payload, dict):
        result.update(report_run_metadata(payload, path))
        evaluation = payload.get("evaluation_result") or payload.get("final_evaluation") or {}
        metric_source = "final"
        if not isinstance(evaluation, dict) or not evaluation:
            attempt_number, attempt_evaluation = best_attempt_evaluation(payload)
            evaluation = attempt_evaluation or {}
            metric_source = f"best attempt {attempt_number}" if attempt_number is not None else "unavailable"
            result["metric_attempt"] = attempt_number
        if isinstance(evaluation, dict) and evaluation:
            result.update(
                {
                    "accuracy": evaluation.get("accuracy"),
                    "syntax_rate": evaluation.get("syntax_rate"),
                    "num_examples": evaluation.get("num_examples"),
                    "num_correct": evaluation.get("num_correct"),
                    "success": evaluation.get("success", payload.get("success")),
                }
            )
            result["metric_source"] = metric_source
        result["total_attempts"] = payload.get("total_attempts")
        result["timestamp"] = payload.get("timestamp")
        result["reported_at"] = payload.get("timestamp") or path.stat().st_mtime
    return result


def latest_reports() -> list[dict]:
    reports: list[Path] = []
    for name in REPORT_GLOBS:
        reports.extend((ROOT_DIR / "outputs" / "generated").rglob(name))
    reports = [path for path in reports if path.is_file()]
    reports.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return [report_summary(path) for path in reports[:20]]


def collect_runtime_alerts(logs: list[dict], *, limit: int = 20) -> list[dict]:
    alerts: list[dict] = []
    for log in logs:
        for alert in log.get("alerts", []):
            alerts.append(
                {
                    **alert,
                    "path": log.get("path"),
                    "mtime": log.get("mtime"),
                }
            )
    alerts.sort(key=lambda item: (item.get("mtime") or 0, item.get("line") or 0), reverse=True)
    return alerts[:limit]


def research_tracker_status() -> dict:
    # Resolve from ROOT_DIR at call time (not the import-time constant) so tests can
    # redirect ROOT_DIR to a fixture; in production ROOT_DIR is the real project dir.
    payload = load_json(ROOT_DIR / "saved-results" / "research_tracker_status.json")
    if not isinstance(payload, dict):
        return {"updated_at": None, "items": []}
    items = payload.get("items")
    if not isinstance(items, list):
        items = []
    return {
        "updated_at": payload.get("updated_at"),
        "items": [item for item in items if isinstance(item, dict)],
    }


def process_tree_owner(pid: str, process_by_pid: dict[str, dict]) -> dict | None:
    current = process_by_pid.get(str(pid))
    seen: set[str] = set()
    best = None
    while current and current["pid"] not in seen:
        seen.add(current["pid"])
        if any(pattern in current["cmd"] for pattern in RUN_PATTERNS):
            best = current
            if "run_all_tests.py" in current["cmd"] or "synthesis.run_synthesis" in current["cmd"]:
                return current
        current = process_by_pid.get(str(current.get("ppid")))
    return best


def queue_status(queue: list[dict], queue_state: object | None) -> tuple[list[dict], dict | None]:
    attempted = set()
    if isinstance(queue_state, dict):
        attempted = {str(item) for item in queue_state.get("attempted_ids", [])}
    pending = [record for record in queue if str(record.get("id")) not in attempted]
    return pending, (pending[0] if pending else None)


def process_status(process: dict, logs: list[dict], reports: list[dict]) -> dict:
    log = log_for_process(process, logs)
    stats = attempt_stats_from_metrics(log.get("recent_metrics", []) if log else [])
    status = "in progress"
    report_path = None
    if log and log.get("status") == "failed":
        status = "failed"
    if log and log.get("status") == "succeeded":
        status = "won"
    output = process.get("output") or ""
    for report in reports:
        path = report.get("path", "")
        if output and output in path:
            report_path = path
            if report.get("kind") == "success" or report.get("success"):
                status = "won"
            elif report.get("kind") == "failure":
                status = "failed"
            break
    prompt_path = prompt_io_for_process(process)
    snippets = rationale_snippets_from_prompt_io(prompt_path) if prompt_path else []
    trend = rationale_trend(snippets, stats)
    return {
        "status": status,
        "log": log,
        "alerts": log.get("alerts", []) if log else [],
        "report_path": report_path,
        "attempt_stats": stats,
        "rationale_trend": trend,
        "prompt_io": str(prompt_path.relative_to(ROOT_DIR)) if prompt_path else None,
    }


def gpu_slots(
    gpus: list[dict],
    gpu_processes: list[dict],
    processes: list[dict],
    queue: list[dict],
    queue_state: object | None,
    logs: list[dict],
    reports: list[dict],
) -> list[dict]:
    process_table = parse_process_table()
    process_by_pid = {process["pid"]: process for process in process_table}
    classified_by_pid = {process["pid"]: classify_run(process) for process in processes}
    pending, next_record = queue_status(queue, queue_state)
    slots: list[dict] = []
    for gpu in gpus:
        uuid = gpu.get("uuid")
        apps = [app for app in gpu_processes if app.get("gpu_uuid") == uuid]
        owners: dict[str, dict] = {}
        for app in apps:
            owner = process_tree_owner(str(app.get("pid")), process_by_pid)
            if owner:
                owners[owner["pid"]] = classify_run(owner)
        owner_details = []
        for process in owners.values():
            owner_details.append({**process, "title": process_title(process), "detail": process_status(process, logs, reports)})
        next_for_gpu = next_record if str(gpu.get("index")) == "3" else None
        if next_for_gpu and owner_details:
            first_cmd = " ".join(str(part) for part in next_for_gpu.get("cmd", []))
            if any((owner.get("cmd") or "") == first_cmd for owner in owner_details):
                next_for_gpu = pending[1] if len(pending) > 1 else None
        slots.append(
            {
                "gpu": gpu,
                "apps": apps,
                "current": owner_details,
                "next": next_for_gpu,
                "pending_count": len(pending) if str(gpu.get("index")) == "3" else 0,
            }
        )
    return slots


def collect_status() -> dict:
    processes = [classify_run(process) for process in parse_processes()]
    gpus = parse_gpu_rows()
    gpu_processes = parse_gpu_processes()
    queue = read_jsonl(QUEUE_PATH)
    queue_state = load_json(STATE_PATH)
    logs = all_recent_logs()
    reports = latest_reports()
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "root": str(ROOT_DIR),
        "gpus": gpus,
        "gpu_processes": gpu_processes,
        "processes": processes,
        "queue": queue,
        "queue_state": queue_state,
        "gpu_slots": gpu_slots(gpus, gpu_processes, processes, queue, queue_state, logs, reports),
        "logs": logs[:10],
        "runtime_alerts": collect_runtime_alerts(logs),
        "research_tracker": research_tracker_status(),
        "reports": reports,
    }


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CSD Experiment Dashboard</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101113;
      --panel: #181a1f;
      --panel-2: #20232a;
      --text: #eceff4;
      --muted: #a9b0bd;
      --line: #313642;
      --green: #4ade80;
      --yellow: #facc15;
      --red: #fb7185;
      --blue: #60a5fa;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      position: sticky;
      top: 0;
      z-index: 2;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 14px 18px;
      border-bottom: 1px solid var(--line);
      background: rgba(16, 17, 19, 0.96);
    }
    h1 { margin: 0; font-size: 18px; font-weight: 650; }
    main { padding: 18px; display: grid; gap: 16px; }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }
    h2 {
      margin: 0;
      padding: 12px 14px;
      font-size: 14px;
      font-weight: 650;
      border-bottom: 1px solid var(--line);
      background: var(--panel-2);
    }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }
    .content { padding: 12px 14px; }
    table { width: 100%; border-collapse: collapse; }
    th, td {
      padding: 8px 10px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }
    th { color: var(--muted); font-size: 12px; font-weight: 600; }
    tr:last-child td { border-bottom: 0; }
    code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    pre {
      margin: 0;
      padding: 12px 14px;
      max-height: 360px;
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
      background: #0b0c0e;
      color: #d8dee9;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 2px 8px;
      border-radius: 999px;
      background: #2a2f39;
      color: var(--text);
      font-size: 12px;
      white-space: nowrap;
    }
    .ok { color: var(--green); }
    .warn { color: var(--yellow); }
    .bad { color: var(--red); }
    tr.ok td { color: var(--green); }
    tr.bad td { color: var(--red); }
    .muted { color: var(--muted); }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; }
    .toolbar { display: flex; align-items: center; gap: 12px; color: var(--muted); }
    .bar { width: 100%; height: 8px; border-radius: 999px; background: #2a2f39; overflow: hidden; }
    .bar > span { display: block; height: 100%; background: var(--blue); }
    .empty { padding: 12px 14px; color: var(--muted); }
    details {
      padding: 8px 0;
      border-bottom: 1px solid var(--line);
    }
    details:last-child { border-bottom: 0; }
    summary { cursor: pointer; font-weight: 600; }
    details[open] summary { margin-bottom: 10px; }
    .detail-grid { display: grid; grid-template-columns: minmax(240px, 1fr) minmax(240px, 1fr); gap: 14px; margin-bottom: 10px; }
    .alert-list, .tracker-list { display: grid; gap: 8px; }
    .alert {
      border-left: 3px solid var(--red);
      padding: 8px 10px;
      background: rgba(251, 113, 133, 0.08);
    }
    .alert .kind { color: var(--red); font-weight: 700; }
    .snippet-title { margin-top: 10px; color: var(--muted); font-size: 12px; font-weight: 650; }
    .tracker-card {
      display: grid;
      gap: 6px;
      padding: 10px 12px;
      border: 1px solid var(--line);
      background: #14161a;
    }
    .tracker-title { font-weight: 700; }
    .tracker-meta { display: flex; flex-wrap: wrap; gap: 8px; }
    ul { margin: 6px 0 10px 18px; padding: 0; }
    li { margin: 3px 0; }
  </style>
</head>
<body>
  <header>
    <h1>CSD Experiment Dashboard</h1>
    <div class="toolbar">
      <span id="updated">Loading...</span>
      <span class="pill">auto-refresh 15s</span>
    </div>
  </header>
  <main>
    <section><h2>GPU Run Slots</h2><div class="content" id="gpu-slots"></div></section>
    <section><h2>Research Collection State</h2><div class="content" id="research-tracker"></div></section>
    <section><h2>Runtime Alerts</h2><div class="content" id="runtime-alerts"></div></section>
    <section><h2>Recent Reports</h2><div class="content" id="reports"></div></section>
    <section><h2>Latest Log Tail</h2><pre id="tail"></pre></section>
  </main>
  <script>
    const esc = value => String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
    const pct = value => Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : (value ?? '?');
    const seconds = value => {
      value = Number(value || 0);
      const h = Math.floor(value / 3600), m = Math.floor((value % 3600) / 60), s = value % 60;
      return h ? `${h}h ${m}m` : (m ? `${m}m ${s}s` : `${s}s`);
    };
    const timeText = value => {
      if (value === null || value === undefined || value === '') return '?';
      const numeric = Number(value);
      const date = Number.isFinite(numeric) ? new Date(numeric * 1000) : new Date(value);
      return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString();
    };
    const table = (headers, rows) => {
      if (!rows.length) return '<div class="empty">No rows.</div>';
      return `<table><thead><tr>${headers.map(h => `<th>${esc(h)}</th>`).join('')}</tr></thead><tbody>${rows.join('')}</tbody></table>`;
    };
    const statusClass = status => status === 'won' ? 'ok' : (status === 'failed' ? 'bad' : 'warn');
    const toneClass = tone => tone === 'ok' ? 'ok' : (tone === 'bad' ? 'bad' : 'warn');
    function attemptLine(stats) {
      const latest = stats?.latest;
      const best = stats?.best_accuracy;
      const bestSyntax = stats?.best_syntax_passing;
      const fmt = item => item ? `#${item.attempt} acc=${item.accuracy_pct ?? '?'}% syntax=${item.syntax_pct ?? '?'}%` : 'none';
      return `<div><b>latest</b>: ${esc(fmt(latest))}</div><div><b>best acc</b>: ${esc(fmt(best))}</div><div><b>best syntax-pass</b>: ${esc(fmt(bestSyntax))}</div>`;
    }
    function processDetails(process) {
      const detail = process.detail || {};
      const trend = detail.rationale_trend || {};
      const logPath = detail.log?.path || '';
      const reasons = (trend.reasons || []).map(r => `<li>${esc(r)}</li>`).join('');
      const snippets = (trend.snippets || []).map(s => `<li class="mono">${esc(s)}</li>`).join('');
      const metricRows = (detail.attempt_stats?.attempts || []).map(a => `<tr><td>${esc(a.attempt)}</td><td>${esc(a.accuracy_pct ?? '')}%</td><td>${esc(a.syntax_pct ?? '')}%</td></tr>`).join('');
      const alerts = (detail.alerts || []).map(a => `<div class="alert"><span class="kind">${esc(a.kind)}</span> <span class="muted">line ${esc(a.line ?? '')}</span><div class="mono">${esc(a.message)}</div></div>`).join('');
      const metadata = [
        process.ablation_sections ? `sections ${process.ablation_sections}` : '',
        process.strategies ? `strategies ${process.strategies}` : '',
        process.max_iterations ? `iter ${process.max_iterations}` : '',
        process.max_steps ? `steps ${process.max_steps}` : '',
        process.eval_sample_size ? `sample ${process.eval_sample_size}` : '',
        process.eval_step_token_budget ? `tb ${process.eval_step_token_budget}` : '',
        process.accuracy_win_margin ? `margin ${process.accuracy_win_margin}` : '',
        process.generated_output_dir ? `out ${process.generated_output_dir}` : '',
        process.ablation_output_dir ? `abl ${process.ablation_output_dir}` : '',
      ].filter(Boolean).join(' · ');
      return `
        <details>
          <summary><span class="${statusClass(detail.status)}">${esc(detail.status || 'in progress')}</span> · ${esc(process.title || process.output || process.cmd)}</summary>
          <div class="detail-grid">
            <div>
              <div class="muted">PID ${esc(process.pid)} · age ${seconds(process.elapsed_seconds)}</div>
              ${metadata ? `<div class="muted">${esc(metadata)}</div>` : ''}
              <div>${attemptLine(detail.attempt_stats || {})}</div>
              <div class="muted mono">log: ${esc(logPath || 'not matched')}</div>
              <div class="muted mono">prompt_io: ${esc(detail.prompt_io || 'not matched')}</div>
            </div>
            <div>
              <div><b>Rationale trend:</b> <span class="${toneClass(trend.tone)}">${esc(trend.label || 'unknown')}</span></div>
              <ul>${reasons || '<li class="muted">No rationale diagnosis yet.</li>'}</ul>
            </div>
          </div>
          ${alerts ? `<div class="snippet-title bad">Runtime alerts for this process</div><div class="alert-list">${alerts}</div>` : ''}
          ${metricRows ? `<table><thead><tr><th>Attempt</th><th>Accuracy</th><th>Syntax</th></tr></thead><tbody>${metricRows}</tbody></table>` : '<div class="empty">No attempt metrics parsed yet.</div>'}
          ${snippets ? `<div class="snippet-title">Recent rationale snippets</div><ul>${snippets}</ul>` : ''}
          <pre>${esc(process.cmd || '')}</pre>
        </details>`;
    }
    function renderGpuSlots(data) {
      const rows = (data.gpu_slots || []).map(slot => {
        const g = slot.gpu || {};
        const used = Number(g.memory_used_mb), total = Number(g.memory_total_mb);
        const width = total ? Math.min(100, (used / total) * 100) : 0;
        const current = (slot.current || []).length
          ? slot.current.map(processDetails).join('')
          : '<div class="empty">No experiment process detected on this GPU.</div>';
        const next = slot.next
          ? `${esc(slot.next.case?.benchmark || '')} · ${esc(slot.next.case?.eval_model || '')} · iter ${esc(slot.next.case?.synth_iter || '')} · steps ${esc(slot.next.case?.max_steps || '')}`
          : 'none queued for this GPU';
        return `<tr>
          <td><span class="pill">GPU ${esc(g.index)}</span><div class="muted">${esc(g.name || '')}</div></td>
          <td>${esc(g.memory_used_mb)} / ${esc(g.memory_total_mb)} MB<div class="bar"><span style="width:${width}%"></span></div><div class="muted">util ${esc(g.utilization_pct)}%</div></td>
          <td>${current}</td>
          <td>${esc(next)}${slot.pending_count ? `<div class="muted">${esc(slot.pending_count)} pending</div>` : ''}</td>
        </tr>`;
      });
      document.getElementById('gpu-slots').innerHTML = table(['GPU', 'Load', 'Current Running Process', 'Per-GPU Queue'], rows);
    }
    function renderLogs(data) {
      const first = data.logs[0];
      document.getElementById('tail').textContent = first ? `${first.path}\n\n${(first.tail || []).join('\n')}` : 'No logs found.';
    }
    function renderRuntimeAlerts(data) {
      const rows = (data.runtime_alerts || []).map(a => `<div class="alert"><span class="kind">${esc(a.kind)}</span> <span class="muted">${esc(timeText(a.mtime))} · ${esc(a.path || '')} · line ${esc(a.line ?? '')}</span><div class="mono">${esc(a.message)}</div></div>`);
      document.getElementById('runtime-alerts').innerHTML = rows.length ? `<div class="alert-list">${rows.join('')}</div>` : '<div class="empty">No runtime alerts in recent logs.</div>';
    }
    function renderResearchTracker(data) {
      const tracker = data.research_tracker || {};
      const rows = (tracker.items || []).map(item => `
        <div class="tracker-card">
          <div class="tracker-meta">
            <span class="pill">${esc(item.state || 'not set')}</span>
            <span class="pill">${esc(item.dataset || 'all datasets')}</span>
          </div>
          <div class="tracker-title">${esc(item.title || 'Untitled update')}</div>
          <div>${esc(item.summary || '')}</div>
          ${item.next ? `<div class="muted"><b>Next:</b> ${esc(item.next)}</div>` : ''}
          ${item.evidence ? `<div class="muted mono">${esc(item.evidence)}</div>` : ''}
        </div>`).join('');
      const updated = tracker.updated_at ? `<div class="muted">Updated ${esc(tracker.updated_at)}</div>` : '';
      document.getElementById('research-tracker').innerHTML = rows
        ? `${updated}<div class="tracker-list">${rows}</div>`
        : '<div class="empty">No research tracker notes found.</div>';
    }
    function renderReports(data) {
      const rows = data.reports.map(r => {
        const klass = r.kind === 'failure' ? 'bad' : (r.kind === 'success' ? 'ok' : 'warn');
        const goal = `acc ${pct(r.min_accuracy)} · syntax ${pct(r.min_syntax_rate)} · delimiters ${esc(r.require_delimiters ?? '')}`;
        const result = `acc ${pct(r.accuracy)} · syntax ${pct(r.syntax_rate)} · ${esc(r.num_correct ?? '')}/${esc(r.num_examples ?? '')}`;
        const run = `iter ${esc(r.max_iterations ?? '')} · backend ${esc(r.eval_backend || '')} · sample ${esc(r.eval_sample_size ?? '')} · steps ${esc(r.eval_max_steps ?? '')} · tb ${esc(r.eval_step_token_budget ?? '')} · sec/ex ${esc(r.eval_max_seconds_per_example ?? '')} · min ${esc(r.min_examples_before_threshold_stop ?? '')}`;
        const author = `${esc(r.author_backend || '')}/${esc(r.author_model || '')} · thinking ${esc(r.author_thinking || '')} · effort ${esc(r.author_effort || '')}`;
        const controls = `policy ${esc(r.helper_selection_policy || '')} · beam ${esc(r.refinement_beam_size ?? '')} · mask ${esc(r.adaptive_helper_mask ?? '')} · restart ${esc(r.restart_after_stuck_iters ?? '')}/${esc(r.restart_cooldown_iters ?? '')}`;
        return `<tr class="${klass}"><td>${esc(r.kind)}</td><td>${esc(timeText(r.reported_at))}</td><td>${esc(r.dataset || '')}</td><td>${esc(r.eval_model || '')}</td><td>${author}</td><td>${goal}</td><td>${result}</td><td>${run}</td><td>${controls}</td><td>${esc(r.metric_source || '')}</td><td class="mono">${esc(r.output_name || '')}</td></tr>`;
      });
      document.getElementById('reports').innerHTML = table(['Kind', 'Run Time', 'Dataset', 'Eval Model', 'Author', 'Goal', 'Result', 'Budget', 'Controls', 'Metric Source', 'Run'], rows);
    }
    async function refresh() {
      try {
        const res = await fetch('/api/status', {cache: 'no-store'});
        const data = await res.json();
        document.getElementById('updated').textContent = `Updated ${data.generated_at}`;
        renderGpuSlots(data);
        renderResearchTracker(data);
        renderRuntimeAlerts(data);
        renderLogs(data);
        renderReports(data);
      } catch (err) {
        document.getElementById('updated').textContent = `Refresh failed: ${err}`;
      }
    }
    refresh();
    setInterval(refresh, 15000);
  </script>
</body>
</html>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        print(f"[dashboard] {self.address_string()} - {format % args}")

    def send_bytes(self, payload: bytes, content_type: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self.send_bytes(HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/api/status":
            payload = json.dumps(collect_status(), sort_keys=True).encode("utf-8")
            self.send_bytes(payload, "application/json; charset=utf-8")
            return
        self.send_error(HTTPStatus.NOT_FOUND, html.escape(path))

    def do_HEAD(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            payload = HTML.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            return
        if path == "/api/status":
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            return
        self.send_error(HTTPStatus.NOT_FOUND, html.escape(path))


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve a live CSD experiment dashboard.")
    parser.add_argument("--host", default=os.environ.get("CSD_DASHBOARD_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("CSD_DASHBOARD_PORT", "8765")))
    return parser


def main() -> int:
    args = make_parser().parse_args()
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"[dashboard] serving http://{args.host}:{args.port} from {ROOT_DIR}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[dashboard] stopping", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
