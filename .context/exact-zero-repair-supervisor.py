#!/usr/bin/env python3
"""Wait for the active cold queue, smoke repaired adapters, then run 31 repairs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


REPO = Path("/home/aadivyar/csd-generation-worktrees/full-baseline-campaign-20260803")
PYTHON = Path("/apps/conda/aadivyar/envs/csd/bin/python")
MANIFEST = REPO / "saved-results/2026-08-04-exact-zero-baseline-repair-manifest.json"
SMOKE_ROOT = REPO / "outputs/baselines/exact-zero-repair-20260804-smoke"
SMOKE_LOG_ROOT = REPO / "logs/exact-zero-repair-20260804-smoke"
GPU_IDS = (0, 2, 3)


def log(message: str) -> None:
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"{stamp} [exact-zero-repair] {message}", flush=True)


def cold_queue_running() -> bool:
    result = subprocess.run(
        [
            "pgrep",
            "-f",
            "scripts.runtime.run_cold_synthesis_queue.*2026-08-03-full-baseline-cold-manifest.json",
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def gpu_memory() -> dict[int, tuple[int, int]]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    values: dict[int, tuple[int, int]] = {}
    for line in output.splitlines():
        index, used, total = (int(part.strip()) for part in line.split(","))
        if index in GPU_IDS:
            values[index] = (used, total)
    return values


def wait_for_gpu(reservation_mib: int) -> int:
    while True:
        memory = gpu_memory()
        candidates = [
            (used, gpu_id)
            for gpu_id, (used, total) in memory.items()
            if used + reservation_mib <= total - 2_000
        ]
        if candidates:
            return min(candidates)[1]
        log(f"waiting for {reservation_mib} MiB capacity on GPUs {GPU_IDS}: {memory}")
        time.sleep(30)


def smoke_command(
    *,
    strategy: str,
    dataset: str,
    model: str,
    output: Path,
    max_steps: int,
    extra: list[str],
) -> list[str]:
    return [
        str(PYTHON),
        "-m",
        "synthesis.evaluate.run_legacy_fixed_strategy",
        "--strategy",
        strategy,
        "--dataset",
        dataset,
        "--eval-model",
        model,
        "--eval-backend",
        "vllm",
        "--device",
        "cuda",
        "--eval-sample-size",
        "1",
        "--eval-max-steps",
        str(max_steps),
        "--eval-step-token-budget",
        "1",
        *extra,
        "--output-json",
        str(output),
    ]


def validate_smoke(path: Path) -> None:
    payload = json.loads(path.read_text())
    answers = payload.get("answers")
    if not isinstance(answers, list) or len(answers) != 1:
        raise RuntimeError(f"{path}: expected one answer")
    generated = str(answers[0].get("generated_answer") or "").strip()
    if not generated:
        raise RuntimeError(f"{path}: generated answer is blank")


def run_smokes() -> None:
    jobs = [
        (
            "spider-qwen35-2b-itergen",
            smoke_command(
                strategy="itergen",
                dataset="spider",
                model="Qwen/Qwen3.5-2B",
                output=SMOKE_ROOT / "spider-qwen35-2b-itergen.json",
                max_steps=176,
                extra=[
                    "--spider-split-file",
                    "environment/benchmark_splits/spider_dev_proportional_300x300_seed334.json",
                    "--spider-split-name",
                    "train",
                ],
            ),
            16_384,
        ),
        *[
            (
                f"smiles-acrylates-qwen25-1p5b-{strategy}",
                smoke_command(
                    strategy=strategy,
                    dataset="smiles",
                    model="Qwen/Qwen2.5-1.5B-Instruct",
                    output=SMOKE_ROOT / f"smiles-acrylates-qwen25-1p5b-{strategy}.json",
                    max_steps=400,
                    extra=[
                        "--smiles-classes",
                        "acrylates",
                        "--smiles-samples-per-class",
                        "1",
                    ],
                ),
                16_000,
            )
            for strategy in ("gcd", "itergen", "crane")
        ],
    ]
    SMOKE_ROOT.mkdir(parents=True, exist_ok=True)
    SMOKE_LOG_ROOT.mkdir(parents=True, exist_ok=True)
    for label, command, reservation in jobs:
        output = Path(command[-1])
        if output.is_file():
            validate_smoke(output)
            log(f"smoke already valid label={label} output={output}")
            continue
        gpu_id = wait_for_gpu(reservation)
        environment = os.environ.copy()
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": str(gpu_id),
                "CSD_HF_KV_CACHE": "0",
                "PYTHONPATH": "synthesis/evaluate:.",
                "PYTHONUNBUFFERED": "1",
            }
        )
        log_path = SMOKE_LOG_ROOT / f"{label}.log"
        log(f"smoke launch label={label} gpu={gpu_id} output={output}")
        with log_path.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                command,
                cwd=REPO,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(
                f"smoke failed label={label} exit={completed.returncode} log={log_path}"
            )
        validate_smoke(output)
        log(f"smoke passed label={label} output={output}")


def run_repairs() -> int:
    payload = json.loads(MANIFEST.read_text())
    labels = [row["label"] for row in payload["rows"]]
    if len(labels) != 31 or len(labels) != len(set(labels)):
        raise RuntimeError(f"expected 31 distinct labels, found {len(labels)}")
    command = [
        str(PYTHON),
        "scripts/run_focal_collection_pool.py",
        "--repo",
        str(REPO),
        "--python",
        str(PYTHON),
        "--campaign",
        "full-baseline-20260803",
        "--campaign-output-name",
        "exact-zero-repair-20260804",
        "--gpu-ids",
        "0,2,3",
        "--max-workers",
        "3",
        "--max-retries",
        "1",
    ]
    for label in labels:
        command.extend(["--include-label", label])
    log(f"repair pool launch cells={len(labels)} gpus={GPU_IDS}")
    return subprocess.run(command, cwd=REPO, check=False).returncode


def main() -> int:
    while cold_queue_running():
        log("waiting for active cold synthesis queue to exit")
        time.sleep(30)
    log("active cold synthesis queue is no longer running")
    run_smokes()
    return run_repairs()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BaseException as error:
        log(f"FAILED: {type(error).__name__}: {error}")
        raise
