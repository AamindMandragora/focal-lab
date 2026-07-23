"""CLI entry for zero-acc babysitter.

Callers: `python -m scripts.runtime.zero_acc_babysitter --local-sim-scenario …`
         `python -m scripts.runtime.zero_acc_babysitter --watch …`
API: main() runs one local scenario or production watch loop.
User instruction: babysitter-implement; launch-full-queue watcher.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from scripts.runtime.zero_acc_babysitter.domains import build_20_cell_ids
from scripts.runtime.zero_acc_babysitter.local_sim import LocalBabysitterSim
from scripts.runtime.zero_acc_babysitter.production_watch import run_watch_loop


def _cell_logs_from_manifest(manifest: Path, exclude_prefixes: list[str]) -> dict[str, Path]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    out: dict[str, Path] = {}
    for job in payload.get("jobs") or []:
        cell = str(job["cell_id"])
        if any(cell.startswith(p) for p in exclude_prefixes):
            continue
        log_file = Path(str(job["log_file"]))
        if not log_file.is_absolute():
            log_file = Path.cwd() / log_file
        out[cell] = log_file
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Zero-acc Cloud babysitter")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--local-sim-scenario",
        choices=[
            "tier_a_harness",
            "tier_a_helper",
            "tier_a_miss",
            "tier_b_helper",
            "tier_b_miss",
            "memory_alone",
            "memory_then_accuracy",
            "telemetry_fail",
            "smoke_retry",
            "acc_ge_15",
        ],
        help="Run one local simulator scenario (never authorizes real queue).",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Run production watch loop over cold-queue cell logs.",
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--exclude-cell-prefix", action="append", default=[])
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    args = parser.parse_args(argv)
    if args.watch:
        if args.manifest is None:
            parser.error("--watch requires --manifest")
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
        cell_logs = _cell_logs_from_manifest(
            args.manifest, [str(p) for p in args.exclude_cell_prefix if str(p)]
        )
        run_watch_loop(
            args.repo_root.resolve(),
            cell_logs,
            poll_seconds=args.poll_seconds,
        )
        return 0
    if not args.local_sim_scenario:
        parser.error("provide --local-sim-scenario or --watch")
    cells = build_20_cell_ids()[:3]
    cell = cells[0]
    scenario = args.local_sim_scenario
    if scenario == "tier_a_harness":
        sim = LocalBabysitterSim(
            repo_root=args.repo_root, cell_ids=cells, twin_accuracy=0.0, helper_failures=[]
        )
        sim.finish_attempt(cell, attempt_index=1, accuracy_pct=0.0)
    elif scenario == "tier_a_helper":
        sim = LocalBabysitterSim(
            repo_root=args.repo_root,
            cell_ids=cells,
            twin_accuracy=0.2,
            helper_failures=["HelperFoo"],
            helper_trace_by_attempt={1: ["HelperFoo"]},
        )
        sim.finish_attempt(cell, attempt_index=1, accuracy_pct=0.0)
    elif scenario == "tier_a_miss":
        sim = LocalBabysitterSim(
            repo_root=args.repo_root, cell_ids=cells, twin_accuracy=0.2, helper_failures=[]
        )
        sim.finish_attempt(cell, attempt_index=1, accuracy_pct=0.0)
    elif scenario == "tier_b_helper":
        sim = LocalBabysitterSim(
            repo_root=args.repo_root,
            cell_ids=cells,
            twin_accuracy=0.2,
            helper_failures=["HelperFoo"],
            helper_trace_by_attempt={1: ["HelperFoo"]},
        )
        sim.finish_attempt(cell, attempt_index=1, accuracy_pct=10.0)
    elif scenario == "tier_b_miss":
        sim = LocalBabysitterSim(
            repo_root=args.repo_root, cell_ids=cells, twin_accuracy=0.2, helper_failures=[]
        )
        sim.finish_attempt(cell, attempt_index=1, accuracy_pct=10.0)
    elif scenario == "memory_alone":
        sim = LocalBabysitterSim(repo_root=args.repo_root, cell_ids=cells)
        sim.abort_memory(cell, attempt_index=1)
    elif scenario == "memory_then_accuracy":
        sim = LocalBabysitterSim(
            repo_root=args.repo_root, cell_ids=cells, twin_accuracy=0.0, helper_failures=[]
        )
        sim.abort_memory_then_finished(cell, attempt_index=1, accuracy_pct=0.0)
    elif scenario == "telemetry_fail":
        sim = LocalBabysitterSim(repo_root=args.repo_root, cell_ids=cells)
        sim.finish_attempt(cell, attempt_index=1, accuracy_pct=None)
    elif scenario == "smoke_retry":
        sim = LocalBabysitterSim(
            repo_root=args.repo_root,
            cell_ids=cells,
            twin_accuracy=0.0,
            helper_failures=[],
            smoke_results=[False, True],
        )
        sim.finish_attempt(cell, attempt_index=1, accuracy_pct=0.0)
    else:  # acc_ge_15
        sim = LocalBabysitterSim(repo_root=args.repo_root, cell_ids=cells)
        sim.finish_attempt(cell, attempt_index=1, accuracy_pct=15.0)
    print(sim.master_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
