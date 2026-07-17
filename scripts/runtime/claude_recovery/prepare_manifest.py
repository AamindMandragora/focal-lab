#!/usr/bin/env python3
"""Order recovery rows so small evaluator jobs can share a GPU safely."""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger("claude-recovery-manifest")
APPROVED_CELLS = {
    "gsm-qwen35-2b",
    "gsm-qwen35-4b",
    "gsm-qwen35-9b",
    "gsm-qwen25-14b",
    "smiles-qwen35-9b-isocyanates",
    "spider-qwen35-4b",
    "spider-qwen25-7b",
}


def validate_approved_cells(jobs: list[dict[str, Any]]) -> None:
    """Refuse protected or accidental rows before the service can start."""
    cells = [str(job.get("cell_id", "")) for job in jobs]
    unapproved = sorted(set(cells) - APPROVED_CELLS)
    if unapproved:
        raise ValueError(f"unapproved recovery cell(s): {', '.join(unapproved)}")
    if len(cells) != len(set(cells)):
        raise ValueError("recovery manifest contains duplicate cell ids")


def ordered_jobs(
    jobs: list[dict[str, Any]], *, gpu_total_mib: int
) -> list[dict[str, Any]]:
    """Honor explicit launch priority, then pack smaller evaluator jobs first."""
    return sorted(
        jobs,
        key=lambda job: (
            int(job.get("dispatch_priority", 1)),
            max(
                int(job["memory_reservation_mib"]),
                math.ceil(float(job["gpu_mem_util"]) * gpu_total_mib),
            ),
            str(job["cell_id"]),
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-total-mib", type=int, default=40960)
    args = parser.parse_args()
    jobs = json.loads(args.source.read_text(encoding="utf-8"))
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("source manifest must be a non-empty list")
    validate_approved_cells(jobs)
    ordered = ordered_jobs(jobs, gpu_total_mib=args.gpu_total_mib)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(ordered, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    LOGGER.warning(
        "[claude-recovery] prepared dispatch order=%s",
        ",".join(str(job["cell_id"]) for job in ordered),
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
    raise SystemExit(main())
