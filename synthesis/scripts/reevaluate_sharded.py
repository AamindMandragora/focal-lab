"""Shard a compiled-CSD re-evaluation across parallel vLLM workers on idle GPUs.

Thin CLI wrapper over synthesis.scripts.sharded_eval_core (the shard/launch/
merge logic lives there so the synthesis loop's per-iteration eval can call
the same code — keep-B-only, one copy of the mechanism).

Usage (from the csd-generation repo root, csd conda env):
  python -m synthesis.scripts.reevaluate_sharded <GeneratedCSD.py> \
      --dataset spider --eval-model ... \
      --spider-split-name test --sample-size 300 --output-json out.json \
      [--workers-per-gpu 2] [--stagger-seconds 45] [--gpu-util 0.18] \
      [--idle-util-threshold 30] [--min-free-mb 8000] [passthrough args...]

The split FILE is hard-coded per dataset (SPLIT_FILE_BY_DATASET in
synthesis/run_constants.py); only the split NAME (train/test) is a flag.

SMILES is not supported (its dataset selection uses --smiles-classes, not an
index split file) — the script refuses rather than guessing.
"""
from __future__ import annotations

import argparse
import sys

from synthesis.run_constants import SPLIT_FILE_BY_DATASET
from synthesis.scripts.sharded_eval_core import run_sharded_reevaluation


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("csd_path")
    p.add_argument("--dataset", required=True)
    p.add_argument("--sample-size", type=int, required=True)
    p.add_argument("--output-json", required=True)
    # Split FILE is hard-coded to the one canonical split per dataset (see
    # SPLIT_FILE_BY_DATASET in synthesis/run_constants.py). Split NAME (which
    # side) stays a required flag (no default) — silently defaulting to one
    # side is how the 2026-07-17 bar/eval split mixup happened.
    p.add_argument("--gsm-split-name", choices=["train", "test"], default=None)
    p.add_argument("--spider-split-name", choices=["train", "test"], default=None)
    p.add_argument("--workers-per-gpu", type=int, default=2)
    p.add_argument("--stagger-seconds", type=int, default=45)
    p.add_argument("--gpu-util", type=float, default=0.18,
                   help="vllm gpu_memory_utilization PER WORKER")
    p.add_argument("--idle-util-threshold", type=int, default=30)
    p.add_argument("--min-free-mb", type=int, default=8000)
    args, passthrough = p.parse_known_args()

    split_file = SPLIT_FILE_BY_DATASET.get(args.dataset)
    split_name = args.spider_split_name if args.dataset == "spider" else args.gsm_split_name
    if not split_file:
        print("[sharded-eval] ERROR: a split file is required for sharding", flush=True)
        return 2
    if not split_name:
        print("[sharded-eval] ERROR: an explicit split name is required when a split file is "
              "given (no default side — see synthesis/split_provenance.py)", flush=True)
        return 2

    return run_sharded_reevaluation(
        csd_path=args.csd_path,
        dataset=args.dataset,
        sample_size=args.sample_size,
        output_json=args.output_json,
        split_file=str(split_file),
        split_name=split_name,
        passthrough=passthrough,
        workers_per_gpu=args.workers_per_gpu,
        stagger_seconds=args.stagger_seconds,
        gpu_util=args.gpu_util,
        idle_util_threshold=args.idle_util_threshold,
        min_free_mb=args.min_free_mb,
    )


if __name__ == "__main__":
    sys.exit(main())
