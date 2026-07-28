#!/usr/bin/env python3
"""Run an approved post-14B rerun manifest exactly once per cell."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import fcntl
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any


APPROVAL_MARKER = "User approval is explicit for post-14B paid Bedrock synthesis"
NO_CAP_APPROVAL_MARKER = (
    "User approval is explicit for all deterministically selected post-14B paid Bedrock synthesis"
)
NO_CAP_APPROVAL_WORDING = (
    "No maximum spend cap; spend only what is necessary for every row selected by the frozen deterministic rule"
)
DETERMINISTIC_RULE_VERSION = "post14b-rebar-v1-accuracy-strict-syntax-clipped-90"
APPROVED_ACCOUNT_ID = "887730490125"
APPROVED_REGION = "us-east-1"
GPU_SAFETY_MIB = 2_000
MANIFEST_FIELDS = [
    "cell_id",
    "dataset",
    "model",
    "class",
    "target_n",
    "min_accuracy_count",
    "min_syntax_count",
    "matrix_sha256",
    "reviewed_json_sha256",
    "evidence_sha256",
    "recipe_json",
]
LEGACY_MANIFEST_FIELDS = [
    field
    for field in MANIFEST_FIELDS
    if field not in {"matrix_sha256", "reviewed_json_sha256", "evidence_sha256"}
]
AUTHOR_SECRET_NAMES = {
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
}
REQUIRED_RECIPE_FIELDS = {
    "output_name",
    "heldout_output_json",
    "gpu",
    "gpu_mem_util",
    "gpu_wait_max_used_mib",
    "train_sample_size",
    "eval_max_steps",
    "eval_max_seconds",
    "train_split_name",
    "heldout_split_name",
    "heldout_sample_size",
}


class ConfigError(ValueError):
    pass


class RunInterrupted(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_detail(value: object) -> str:
    return str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ")


def row_sha256(row: dict[str, Any]) -> str:
    """Hash the executable meaning of one manifest row."""
    payload = {
        key: row[key]
        for key in (
            "cell_id",
            "dataset",
            "model",
            "class",
            "target_n",
            "min_accuracy_count",
            "min_syntax_count",
            "matrix_sha256",
            "reviewed_json_sha256",
            "evidence_sha256",
        )
    }
    payload["recipe"] = row["recipe"]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def append_state(
    path: Path,
    *,
    cell_id: str,
    status: str,
    manifest_sha256: str,
    detail: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write("timestamp\tcell_id\tstatus\tmanifest_sha256\tdetail\n")
        handle.write(
            "\t".join(
                [
                    utc_now(),
                    safe_detail(cell_id),
                    safe_detail(status),
                    manifest_sha256,
                    safe_detail(detail),
                ]
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def int_field(row: dict[str, str], name: str) -> int:
    try:
        value = int(row[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigError(f"invalid integer {name!r} for cell {row.get('cell_id')!r}") from exc
    return value


def load_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ConfigError(f"manifest missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames not in (MANIFEST_FIELDS, LEGACY_MANIFEST_FIELDS):
            raise ConfigError(
                "manifest columns must match the current evidence-bound contract or "
                f"the legacy capped-approval contract, got {reader.fieldnames}"
            )
        evidence_bound = reader.fieldnames == MANIFEST_FIELDS
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw in reader:
            cell_id = raw["cell_id"].strip()
            if not cell_id or cell_id in seen:
                raise ConfigError(f"empty or duplicate cell_id: {cell_id!r}")
            seen.add(cell_id)
            dataset = raw["dataset"].strip()
            if dataset not in {"gsm_symbolic", "spider", "smiles"}:
                raise ConfigError(f"unsupported dataset for {cell_id}: {dataset!r}")
            target_n = int_field(raw, "target_n")
            min_accuracy_count = int_field(raw, "min_accuracy_count")
            min_syntax_count = int_field(raw, "min_syntax_count")
            if target_n <= 0:
                raise ConfigError(f"target_n must be positive for {cell_id}")
            if not 0 <= min_accuracy_count <= target_n:
                raise ConfigError(f"accuracy count out of range for {cell_id}")
            if not 0 <= min_syntax_count <= target_n:
                raise ConfigError(f"syntax count out of range for {cell_id}")
            try:
                recipe = json.loads(raw["recipe_json"])
            except json.JSONDecodeError as exc:
                raise ConfigError(f"invalid recipe_json for {cell_id}: {exc}") from exc
            if not isinstance(recipe, dict):
                raise ConfigError(f"recipe_json must be an object for {cell_id}")
            missing = sorted(REQUIRED_RECIPE_FIELDS - recipe.keys())
            if missing:
                raise ConfigError(f"recipe for {cell_id} is missing {missing}")
            if recipe.get("cold") is not True:
                raise ConfigError(f"recipe for {cell_id} must explicitly record cold=true")
            if evidence_bound:
                for hash_field in ("matrix_sha256", "reviewed_json_sha256", "evidence_sha256"):
                    if not re.fullmatch(r"[0-9a-f]{64}", raw.get(hash_field, "")):
                        raise ConfigError(f"invalid {hash_field} for cell {cell_id}")
            serialized = json.dumps(recipe, sort_keys=True)
            if "initial_strategy" in serialized or "--initial-strategy-file" in serialized:
                raise ConfigError(f"warm-start field is forbidden for synthesis cell {cell_id}")
            rows.append(
                {
                    **raw,
                    "cell_id": cell_id,
                    "dataset": dataset,
                    "target_n": target_n,
                    "min_accuracy_count": min_accuracy_count,
                    "min_syntax_count": min_syntax_count,
                    "recipe": recipe,
                }
            )
    if not rows:
        raise ConfigError("manifest has no jobs")
    return rows


def load_and_validate_approval(
    path: Path,
    *,
    matrix_sha256: str,
    snapshot_sha256: str,
    manifest_sha256: str,
    audit_sha256: str,
    run_synth_sha256: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"approval missing: {path}")
    try:
        approval = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ConfigError(f"invalid approval JSON: {exc}") from exc
    no_cap = approval.get("max_approved_cost_usd") is None
    expected_marker = NO_CAP_APPROVAL_MARKER if no_cap else APPROVAL_MARKER
    if approval.get("approval_marker") != expected_marker:
        raise ConfigError("approval marker is missing or incorrect")
    if approval.get("matrix_sha256") != matrix_sha256:
        raise ConfigError("approval matrix SHA-256 does not match the frozen matrix")
    if approval.get("pre_rebar_snapshot_sha256") != snapshot_sha256:
        raise ConfigError("approval pre-rebar snapshot SHA-256 does not match")
    if approval.get("manifest_sha256") != manifest_sha256:
        raise ConfigError("approval manifest SHA-256 does not match the executable manifest")
    if approval.get("audit_sha256") != audit_sha256:
        raise ConfigError("approval audit SHA-256 does not match the reviewed scanner audit")
    if approval.get("run_synth_sha256") != run_synth_sha256:
        raise ConfigError("approval synthesis launcher SHA-256 does not match --run-synth")
    cell_ids = [row["cell_id"] for row in rows]
    if approval.get("cell_count") != len(cell_ids):
        raise ConfigError("approval cell_count does not match the manifest")
    if sorted(approval.get("approved_cells", [])) != sorted(cell_ids):
        raise ConfigError("approval approved_cells do not match the manifest")
    account_id = str(approval.get("account_id", "")).strip()
    if not re.fullmatch(r"\d{12}", account_id):
        raise ConfigError("approval must name the literal 12-digit AWS account id")
    region = str(approval.get("region", "")).strip()
    if not re.fullmatch(r"[a-z]{2}(?:-gov)?-[a-z]+-\d+", region):
        raise ConfigError("approval must name a concrete AWS billing region")
    if no_cap:
        if account_id != APPROVED_ACCOUNT_ID or region != APPROVED_REGION:
            raise ConfigError("no-cap approval is valid only for the recorded AWS account and region")
        if approval.get("approval_scope") != NO_CAP_APPROVAL_WORDING:
            raise ConfigError("no-cap approval must contain the exact explicit no-cap scope")
        if approval.get("deterministic_rule_version") != DETERMINISTIC_RULE_VERSION:
            raise ConfigError("no-cap approval deterministic rule version does not match")
        expected_row_hashes = {row["cell_id"]: row_sha256(row) for row in rows}
        if approval.get("approved_row_sha256") != expected_row_hashes:
            raise ConfigError("no-cap approval row hashes do not match the manifest rows")
    try:
        estimated_min = float(approval["estimated_cost_min_usd"])
        estimated_max = float(approval["estimated_cost_max_usd"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigError("approval estimated cost fields must be numeric") from exc
    approved_max = None
    if not no_cap:
        try:
            approved_max = float(approval["max_approved_cost_usd"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigError("approval maximum cost must be numeric or null") from exc
    finite_values = (estimated_min, estimated_max) if no_cap else (estimated_min, estimated_max, approved_max)
    if not all(math.isfinite(value) for value in finite_values):
        raise ConfigError("approval cost fields must be finite numbers")
    if estimated_min < 0 or estimated_max < estimated_min:
        raise ConfigError("approval cost range or maximum is invalid")
    if approved_max is not None and approved_max < estimated_max:
        raise ConfigError("approval cost range or maximum is invalid")
    return approval


def load_and_validate_audit(
    path: Path,
    *,
    matrix_sha256: str,
    snapshot_sha256: str,
    manifest_sha256: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"scanner audit missing: {path}")
    try:
        audit = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ConfigError(f"invalid scanner audit JSON: {exc}") from exc
    if not isinstance(audit, dict):
        raise ConfigError("scanner audit must be a JSON object")
    if audit.get("verdict") != "ok" or audit.get("errors") != []:
        raise ConfigError("scanner audit is not an error-free reviewed audit")
    if audit.get("matrix_sha256") != matrix_sha256:
        raise ConfigError("scanner audit matrix SHA-256 does not match the frozen matrix")
    if audit.get("pre_rebar_snapshot_sha256") != snapshot_sha256:
        raise ConfigError("scanner audit pre-rebar snapshot SHA-256 does not match")
    if audit.get("candidate_manifest_sha256") != manifest_sha256:
        raise ConfigError("scanner audit is not bound to the executable manifest SHA-256")
    expected_cells = sorted(row["cell_id"] for row in rows)
    if sorted(audit.get("candidate_cells", [])) != expected_cells:
        raise ConfigError("scanner audit candidate cells do not match the manifest")
    return audit


def load_and_validate_snapshot(path: Path, matrix_sha256: str) -> str:
    if not path.is_file():
        raise ConfigError(f"pre-rebar snapshot missing: {path}")
    try:
        snapshot = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ConfigError(f"invalid pre-rebar snapshot JSON: {exc}") from exc
    if not isinstance(snapshot, dict) or snapshot.get("state") != "complete":
        raise ConfigError("pre-rebar snapshot is not complete")
    if snapshot.get("matrix_sha256") != matrix_sha256:
        raise ConfigError("pre-rebar snapshot matrix SHA-256 does not match")
    return sha256_file(path)


def decimal_ratio(count: int, total: int) -> str:
    return f"{count / total:.12g}"


def require_recipe_number(recipe: dict[str, Any], name: str, cast):
    try:
        return cast(recipe[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigError(f"invalid recipe field {name!r}") from exc


def synthesis_environment(
    row: dict[str, Any], inherited: dict[str, str], assigned_gpu: int | None = None
) -> dict[str, str]:
    recipe = row["recipe"]
    env = dict(inherited)
    env.update(
        {
            "DATASET": row["dataset"],
            "EVAL_MODEL": row["model"],
            "GPU": str(
                assigned_gpu
                if assigned_gpu is not None
                else require_recipe_number(recipe, "gpu", int)
            ),
            "GPU_MEM_UTIL": str(require_recipe_number(recipe, "gpu_mem_util", float)),
            "MAX_ITERS": "40",
            "SAMPLE_SIZE": str(require_recipe_number(recipe, "train_sample_size", int)),
            "MIN_ACC": decimal_ratio(row["min_accuracy_count"], row["target_n"]),
            "MIN_SYN": decimal_ratio(row["min_syntax_count"], row["target_n"]),
            "EVAL_MAX_STEPS": str(require_recipe_number(recipe, "eval_max_steps", int)),
            "EVAL_MAX_SECONDS": str(require_recipe_number(recipe, "eval_max_seconds", int)),
            "OUTPUT_NAME": str(recipe["output_name"]),
            "SPLIT_NAME": str(recipe["train_split_name"]),
        }
    )
    if recipe.get("train_split_file"):
        env["SPLIT_FILE"] = str(recipe["train_split_file"])
    if row["dataset"] == "smiles":
        if not row["class"]:
            raise ConfigError(f"SMILES cell {row['cell_id']} has no class")
        if not recipe.get("smiles_task"):
            raise ConfigError(f"SMILES cell {row['cell_id']} has no smiles_task")
        env["SMILES_CLASS"] = row["class"]
        env["SMILES_TASK"] = str(recipe["smiles_task"])
    return env


def display_synthesis_command(env: dict[str, str], run_synth: Path) -> str:
    keys = [
        "DATASET",
        "EVAL_MODEL",
        "GPU",
        "GPU_MEM_UTIL",
        "MAX_ITERS",
        "SAMPLE_SIZE",
        "MIN_ACC",
        "MIN_SYN",
        "EVAL_MAX_STEPS",
        "EVAL_MAX_SECONDS",
        "OUTPUT_NAME",
        "SPLIT_NAME",
        "SPLIT_FILE",
        "SMILES_CLASS",
    ]
    assignments = " ".join(f"{key}={env[key]}" for key in keys if key in env)
    return f"env {assignments} bash {run_synth}"


def compiled_csd_for_run(repo: Path, output_name: str) -> Path | None:
    latest = repo / "outputs" / "generated" / output_name / "latest_run.txt"
    if not latest.is_file():
        return None
    run_dir = Path(latest.read_text(encoding="utf-8").strip())
    if not run_dir.is_absolute():
        run_dir = repo / run_dir
    report = run_dir / "results" / "success_report.json"
    if not report.is_file():
        return None
    try:
        payload = json.loads(report.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    compiled_dir = Path(str(payload.get("compiled_dir", "")))
    if not compiled_dir.is_absolute():
        compiled_dir = repo / compiled_dir
    csd = compiled_dir / "GeneratedCSD.py"
    return csd if csd.is_file() else None


def heldout_command(row: dict[str, Any], python: Path, csd: Path) -> list[str]:
    recipe = row["recipe"]
    command = [
        str(python),
        "-m",
        "synthesis.scripts.reevaluate_compiled_csd",
        str(csd),
        "--dataset",
        row["dataset"],
        "--eval-model",
        row["model"],
        "--eval-backend",
        "vllm",
        "--device",
        "auto",
        "--sample-size",
        str(require_recipe_number(recipe, "heldout_sample_size", int)),
        "--max-steps",
        str(require_recipe_number(recipe, "eval_max_steps", int)),
        "--step-token-budget",
        str(int(recipe.get("step_token_budget", 1))),
        "--max-seconds-per-example",
        str(require_recipe_number(recipe, "eval_max_seconds", int)),
        "--vllm-gpu-memory-utilization",
        str(require_recipe_number(recipe, "gpu_mem_util", float)),
        "--vllm-tensor-parallel-size",
        "1",
        "--output-json",
        str(recipe["heldout_output_json"]),
    ]
    split_name = str(recipe["heldout_split_name"])
    split_file = recipe.get("heldout_split_file")
    if row["dataset"] == "gsm_symbolic":
        if split_file:
            command.extend(["--gsm-split-file", str(split_file)])
        command.extend(["--gsm-split-name", split_name])
    elif row["dataset"] == "spider":
        if split_file:
            command.extend(["--spider-split-file", str(split_file)])
        command.extend(["--spider-split-name", split_name])
    else:
        command.extend(["--smiles-classes", row["class"]])
    return command


def author_free_environment(inherited: dict[str, str], gpu: object) -> dict[str, str]:
    clean = {}
    for key, value in inherited.items():
        if key.startswith("AWS_") or key.startswith("BEDROCK_"):
            continue
        if key in AUTHOR_SECRET_NAMES or key.endswith("_API_KEY"):
            continue
        clean[key] = value
    clean["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return clean


def exact_count(payload: dict[str, Any], *, metric: str, total: int) -> int | None:
    count_names = {
        "accuracy": ("num_correct", "correct_count"),
        "syntax_rate": ("num_syntax_valid", "syntax_valid_count"),
    }
    for name in count_names[metric]:
        value = payload.get(name)
        if isinstance(value, int) and 0 <= value <= total:
            return value
    value = payload.get(metric)
    if not isinstance(value, (int, float)):
        return None
    scaled = float(value) * total
    rounded = round(scaled)
    return int(rounded) if math.isclose(scaled, rounded, abs_tol=1e-7) else None


def claim_directory(claims_dir: Path, cell_id: str) -> Path:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", cell_id).strip("_")[:80] or "cell"
    suffix = hashlib.sha256(cell_id.encode("utf-8")).hexdigest()[:12]
    return claims_dir / f"{slug}-{suffix}"


def claim_cell(claims_dir: Path, cell_id: str, manifest_sha256: str) -> bool:
    claims_dir.mkdir(parents=True, exist_ok=True)
    claim_dir = claim_directory(claims_dir, cell_id)
    try:
        claim_dir.mkdir()
    except FileExistsError:
        return False
    (claim_dir / "claim.json").write_text(
        json.dumps(
            {
                "cell_id": cell_id,
                "manifest_sha256": manifest_sha256,
                "started_at": utc_now(),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return True


def log(message: str) -> None:
    print(f"[post14b-rebar] {message}", flush=True)


def gpu_memory_used_mib(nvidia_smi: str, gpu: object) -> int:
    result = subprocess.run(
        [
            nvidia_smi,
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
            "-i",
            str(gpu),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = safe_detail(result.stderr or result.stdout or f"exit={result.returncode}")
        raise ConfigError(f"GPU status command failed: {detail}")
    first_line = result.stdout.strip().splitlines()
    if not first_line:
        raise ConfigError("GPU status command failed: empty memory output")
    try:
        return int(first_line[0].strip())
    except ValueError as exc:
        raise ConfigError(
            f"GPU status command failed: invalid memory output {first_line[0]!r}"
        ) from exc


def gpu_memory_snapshot(nvidia_smi: str) -> dict[int, dict[str, int]]:
    result = subprocess.run(
        [
            nvidia_smi,
            "--query-gpu=index,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = safe_detail(result.stderr or result.stdout or f"exit={result.returncode}")
        raise ConfigError(f"GPU status command failed: {detail}")
    snapshots: dict[int, dict[str, int]] = {}
    try:
        for fallback_gpu, line in enumerate(result.stdout.strip().splitlines()):
            parts = [int(part.strip()) for part in line.split(",")]
            if len(parts) == 1:  # compatibility with simple test/status shims
                gpu, used, total = fallback_gpu, parts[0], 40_960
            elif len(parts) == 3:
                gpu, used, total = parts
            else:
                raise ValueError(line)
            snapshots[gpu] = {"used_mib": used, "total_mib": total}
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"GPU status command failed: invalid snapshot {result.stdout!r}") from exc
    if not snapshots:
        raise ConfigError("GPU status command failed: empty snapshot")
    return snapshots


def memory_reservation_mib(row: dict[str, Any]) -> int:
    value = row["recipe"].get("memory_reservation_mib")
    if value is None:
        value = 32_768 if "14b" in row["model"].lower() else 18_432
    try:
        reservation = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"invalid memory reservation for {row['cell_id']}") from exc
    if reservation <= 0:
        raise ConfigError(f"memory reservation must be positive for {row['cell_id']}")
    return reservation


def choose_gpu(
    row: dict[str, Any],
    snapshots: dict[int, dict[str, int]],
    reservations: dict[int, dict[str, int]],
    baseline_snapshots: dict[int, dict[str, int]] | None = None,
) -> int | None:
    required = memory_reservation_mib(row)
    candidates: list[tuple[int, int]] = []
    for gpu, snapshot in snapshots.items():
        reserved = sum(reservations.get(gpu, {}).values())
        baseline_used = (
            baseline_snapshots.get(gpu, snapshot)["used_mib"]
            if baseline_snapshots is not None
            else snapshot["used_mib"]
        )
        projected_used = max(snapshot["used_mib"], baseline_used + reserved)
        projected = projected_used + required
        safe_capacity = snapshot["total_mib"] - GPU_SAFETY_MIB
        if projected <= safe_capacity:
            candidates.append((projected_used, gpu))
    return min(candidates)[1] if candidates else None


def dispatch_rows(
    *,
    rows: list[dict[str, Any]],
    gpu_snapshot,
    worker,
    poll_seconds: float,
) -> None:
    """Greedily run rows while retaining each reservation until its worker exits."""
    pending = list(rows)
    reservations: dict[int, dict[str, int]] = {}
    running: dict[concurrent.futures.Future, tuple[int, str]] = {}
    baseline_snapshots = gpu_snapshot()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(rows))) as executor:
        while pending or running:
            snapshots = gpu_snapshot()
            for gpu in snapshots:
                reservations.setdefault(gpu, {})
                if not reservations[gpu]:
                    baseline_snapshots[gpu] = dict(snapshots[gpu])
            launched = True
            while pending and launched:
                launched = False
                for index, row in enumerate(pending):
                    gpu = choose_gpu(
                        row, snapshots, reservations, baseline_snapshots
                    )
                    if gpu is None:
                        continue
                    cell_id = row["cell_id"]
                    reservation = memory_reservation_mib(row)
                    reservations[gpu][cell_id] = reservation
                    log(
                        f"dispatch cell={cell_id} gpu={gpu} reservation_mib={reservation} "
                        f"reserved_total_mib={sum(reservations[gpu].values())}"
                    )
                    future = executor.submit(worker, row, gpu)
                    running[future] = (gpu, cell_id)
                    pending.pop(index)
                    launched = True
                    break
            finished = [future for future in running if future.done()]
            for future in finished:
                gpu, cell_id = running.pop(future)
                reservations[gpu].pop(cell_id, None)
                future.result()
                log(f"release cell={cell_id} gpu={gpu}")
            if finished:
                continue
            if pending and not running:
                raise ConfigError("no pending job fits current GPU memory with the safety reserve")
            if pending or running:
                time.sleep(max(0.0, poll_seconds))


def wait_for_gpu(
    *, recipe: dict[str, Any], nvidia_smi: str, poll_seconds: float, cell_id: str
) -> None:
    gpu = recipe["gpu"]
    maximum = require_recipe_number(recipe, "gpu_wait_max_used_mib", int)
    if maximum < 0:
        raise ConfigError("gpu_wait_max_used_mib must be nonnegative")
    while True:
        used = gpu_memory_used_mib(nvidia_smi, gpu)
        if used <= maximum:
            log(f"GPU ready cell={cell_id} gpu={gpu} used_mib={used} limit_mib={maximum}")
            return
        log(f"waiting cell={cell_id} gpu={gpu} used_mib={used} limit_mib={maximum}")
        if poll_seconds <= 0:
            raise ConfigError(
                f"GPU {gpu} uses {used} MiB above limit {maximum} MiB and polling is disabled"
            )
        time.sleep(poll_seconds)


def run_row(
    row: dict[str, Any],
    *,
    repo: Path,
    state: Path,
    claims_dir: Path,
    manifest_sha256: str,
    run_synth: Path,
    python: Path,
    nvidia_smi: str,
    gpu_wait_poll_seconds: float,
    dry_run: bool,
    assigned_gpu: int | None = None,
) -> None:
    cell_id = row["cell_id"]
    gpu = (
        assigned_gpu
        if assigned_gpu is not None
        else require_recipe_number(row["recipe"], "gpu", int)
    )
    env = synthesis_environment(row, dict(os.environ), assigned_gpu=gpu)
    synth_display = display_synthesis_command(env, run_synth)
    placeholder = Path("<compiled-csd>")
    heldout_dry = heldout_command(row, python, placeholder)
    if dry_run:
        log(f"DRY_RUN cell={cell_id} synthesis: {synth_display}")
        log(f"DRY_RUN cell={cell_id} heldout: {' '.join(heldout_dry)}")
        return

    if claim_directory(claims_dir, cell_id).exists():
        log(f"skip cell={cell_id}: already claimed")
        return

    if assigned_gpu is None:
        wait_for_gpu(
            recipe=row["recipe"],
            nvidia_smi=nvidia_smi,
            poll_seconds=gpu_wait_poll_seconds,
            cell_id=cell_id,
        )
    if not claim_cell(claims_dir, cell_id, manifest_sha256):
        log(f"skip cell={cell_id}: already claimed")
        return
    append_state(
        state,
        cell_id=cell_id,
        status="started",
        manifest_sha256=manifest_sha256,
        detail=f"output_name={row['recipe']['output_name']}",
    )
    log(f"start cell={cell_id} target={row['min_accuracy_count']}/{row['target_n']} acc, {row['min_syntax_count']}/{row['target_n']} syntax")
    try:
        synth = subprocess.run(["bash", str(run_synth)], cwd=repo, env=env, check=False)
        if synth.returncode != 0:
            append_state(
                state,
                cell_id=cell_id,
                status="synthesis_error",
                manifest_sha256=manifest_sha256,
                detail=f"exit={synth.returncode}",
            )
            log(f"finish cell={cell_id} synthesis_error exit={synth.returncode}")
            return
        csd = compiled_csd_for_run(repo, str(row["recipe"]["output_name"]))
        if csd is None:
            append_state(
                state,
                cell_id=cell_id,
                status="no_accept",
                manifest_sha256=manifest_sha256,
                detail="no compiled accepted CSD",
            )
            log(f"finish cell={cell_id} no_accept")
            return
        command = heldout_command(row, python, csd)
        output_json = Path(str(row["recipe"]["heldout_output_json"]))
        if not output_json.is_absolute():
            output_json = repo / output_json
        output_json.parent.mkdir(parents=True, exist_ok=True)
        heldout_env = author_free_environment(os.environ, gpu)
        log(f"heldout cell={cell_id} evaluator=synthesis.scripts.reevaluate_compiled_csd")
        heldout = subprocess.run(command, cwd=repo, env=heldout_env, check=False)
        if heldout.returncode != 0 or not output_json.is_file():
            append_state(
                state,
                cell_id=cell_id,
                status="heldout_loss",
                manifest_sha256=manifest_sha256,
                detail=f"reeval_exit={heldout.returncode}; output_exists={output_json.is_file()}",
            )
            log(f"finish cell={cell_id} heldout_loss reeval_exit={heldout.returncode}")
            return
        payload = json.loads(output_json.read_text(encoding="utf-8"))
        total = payload.get("num_examples")
        if total != row["target_n"]:
            append_state(
                state,
                cell_id=cell_id,
                status="heldout_loss",
                manifest_sha256=manifest_sha256,
                detail=f"unexpected num_examples={total}; expected={row['target_n']}",
            )
            log(f"finish cell={cell_id} heldout_loss wrong sample count")
            return
        accuracy_count = exact_count(payload, metric="accuracy", total=total)
        syntax_count = exact_count(payload, metric="syntax_rate", total=total)
        won = (
            accuracy_count is not None
            and syntax_count is not None
            and accuracy_count >= row["min_accuracy_count"]
            and syntax_count >= row["min_syntax_count"]
        )
        status = "heldout_win" if won else "heldout_loss"
        append_state(
            state,
            cell_id=cell_id,
            status=status,
            manifest_sha256=manifest_sha256,
            detail=f"accuracy_count={accuracy_count}; syntax_count={syntax_count}; n={total}",
        )
        log(f"finish cell={cell_id} {status} acc={accuracy_count}/{total} syntax={syntax_count}/{total}")
    except RunInterrupted:
        append_state(
            state,
            cell_id=cell_id,
            status="interrupted",
            manifest_sha256=manifest_sha256,
            detail="runner received termination signal",
        )
        log(f"finish cell={cell_id} interrupted")
    except Exception as exc:  # preserve the one-cycle claim on unexpected failures
        append_state(
            state,
            cell_id=cell_id,
            status="synthesis_error",
            manifest_sha256=manifest_sha256,
            detail=f"unexpected {type(exc).__name__}: {exc}",
        )
        log(f"finish cell={cell_id} unexpected_error={type(exc).__name__}: {exc}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--claims-dir", type=Path, required=True)
    parser.add_argument("--run-synth", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    parser.add_argument("--gpu-wait-poll-seconds", type=float, default=300.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        repo = args.repo.resolve()
        matrix = args.matrix.resolve()
        snapshot_path = args.snapshot.resolve()
        manifest = args.manifest.resolve()
        audit_path = args.audit.resolve()
        approval_path = args.approval.resolve()
        run_synth = args.run_synth.resolve()
        if not repo.is_dir():
            raise ConfigError(f"repo missing: {repo}")
        if not matrix.is_file():
            raise ConfigError(f"frozen matrix missing: {matrix}")
        if not run_synth.is_file():
            raise ConfigError(f"synthesis launcher missing: {run_synth}")
        rows = load_manifest(manifest)
        matrix_sha256 = sha256_file(matrix)
        snapshot_sha256 = load_and_validate_snapshot(snapshot_path, matrix_sha256)
        manifest_sha256 = sha256_file(manifest)
        audit = load_and_validate_audit(
            audit_path,
            matrix_sha256=matrix_sha256,
            snapshot_sha256=snapshot_sha256,
            manifest_sha256=manifest_sha256,
            rows=rows,
        )
        audit_sha256 = sha256_file(audit_path)
        run_synth_sha256 = sha256_file(run_synth)
        approval = load_and_validate_approval(
            approval_path,
            matrix_sha256=matrix_sha256,
            snapshot_sha256=snapshot_sha256,
            manifest_sha256=manifest_sha256,
            audit_sha256=audit_sha256,
            run_synth_sha256=run_synth_sha256,
            rows=rows,
        )
        log(
            "validated inputs "
            f"cells={len(rows)} matrix_sha={matrix_sha256[:12]} snapshot_sha={snapshot_sha256[:12]} "
            f"manifest_sha={manifest_sha256[:12]} "
            f"audit_sha={audit_sha256[:12]} run_synth_sha={run_synth_sha256[:12]} "
            f"account={approval['account_id']} region={approval['region']} max_cost_usd={approval['max_approved_cost_usd']}"
        )

        def interrupted(_signum, _frame):
            raise RunInterrupted()

        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        worker_args = {
            "repo": repo,
            "state": args.state.resolve(),
            "claims_dir": args.claims_dir.resolve(),
            "manifest_sha256": manifest_sha256,
            "run_synth": run_synth,
            "python": args.python.resolve(),
            "nvidia_smi": args.nvidia_smi,
            "gpu_wait_poll_seconds": args.gpu_wait_poll_seconds,
            "dry_run": args.dry_run,
        }
        schedulable_rows = []
        for row in rows:
            if claim_directory(args.claims_dir.resolve(), row["cell_id"]).exists():
                log(f"skip cell={row['cell_id']}: already claimed")
            else:
                schedulable_rows.append(row)
        if args.dry_run:
            for row in schedulable_rows:
                run_row(row, **worker_args)
        elif schedulable_rows:
            dispatch_rows(
                rows=schedulable_rows,
                gpu_snapshot=lambda: gpu_memory_snapshot(args.nvidia_smi),
                worker=lambda row, gpu: run_row(
                    row, assigned_gpu=gpu, **worker_args
                ),
                poll_seconds=args.gpu_wait_poll_seconds,
            )
        return 0
    except ConfigError as exc:
        print(f"[post14b-rebar] configuration error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
