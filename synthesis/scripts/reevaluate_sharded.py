"""Shard a compiled-CSD re-evaluation across parallel vLLM workers on idle GPUs.

Wraps synthesis.scripts.reevaluate_compiled_csd (unchanged): slices the split
file into contiguous shards, launches one worker process per shard with
staggered starts (concurrent vLLM engine init races on memory profiling —
2026-07-17 mini test), waits, then merges the part JSONs into one result with
recomputed accuracy/syntax_rate. Trajectories are batch-of-1, identical to a
sequential run (validated: 40/40 is_correct match, minitest saved-results).

Usage (from the csd-generation repo root, csd conda env):
  python -m synthesis.scripts.reevaluate_sharded <GeneratedCSD.py> \
      --dataset spider --eval-model ... --spider-split-file ... \
      --spider-split-name test --sample-size 300 --output-json out.json \
      [--workers-per-gpu 2] [--stagger-seconds 45] [--gpu-util 0.18] \
      [--idle-util-threshold 30] [--min-free-mb 8000] [passthrough args...]

SMILES is not supported (its dataset selection uses --smiles-classes, not an
index split file) — the script refuses rather than guessing.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

LOG = "[sharded-eval]"

# dataset -> split-name -> key in the split JSON holding example indices.
# One vocabulary per dataset, no aliases: Spider's held-out side is "test"
# (the "eval" alias was removed 2026-07-17 — aliases hid a bar/eval split
# mixup). Note: the old spider "eval" -> eval_indices mapping is gone; in the
# probe/oracle manifests eval_indices differed from test_indices, so if you
# need those 300-example sets, use the proportional 300x300 manifest instead.
_INDICES_KEYS = {
    "spider": {"train": "train_indices", "test": "test_indices"},
    "gsm_symbolic": {"train": "train_indices", "test": "test_indices"},
}


def indices_key_for(dataset: str, split_name: str) -> str:
    if dataset not in _INDICES_KEYS:
        raise ValueError(
            f"dataset {dataset!r} not shardable (no index-based split file); "
            f"supported: {sorted(_INDICES_KEYS)}"
        )
    keys = _INDICES_KEYS[dataset]
    if split_name not in keys:
        raise ValueError(f"split name {split_name!r} invalid for {dataset}: {sorted(keys)}")
    return keys[split_name]


def plan_shards(n_examples: int, n_workers: int) -> list[tuple[int, int]]:
    """Contiguous (lo, hi) slices covering n_examples, sizes differing by <=1."""
    n_workers = min(n_workers, n_examples)
    base, extra = divmod(n_examples, n_workers)
    shards, lo = [], 0
    for i in range(n_workers):
        hi = lo + base + (1 if i < extra else 0)
        shards.append((lo, hi))
        lo = hi
    return shards


def slice_split(split: dict, indices_key: str, lo: int, hi: int) -> dict:
    out = dict(split)
    out[indices_key] = split[indices_key][lo:hi]
    size_key = indices_key.replace("_indices", "_size")
    if size_key in out:
        out[size_key] = hi - lo
    return out


def merge_results(parts: list[dict]) -> dict:
    if not parts:
        raise ValueError("no shard results to merge")
    answers = [a for p in parts for a in p["answers"]]
    n = len(answers)
    merged_metrics = {
        "num_shards": len(parts),
        "shard_sizes": [len(p["answers"]) for p in parts],
    }
    return {
        "accuracy": sum(bool(a["is_correct"]) for a in answers) / n,
        "syntax_rate": sum(bool(a["is_syntax_valid"]) for a in answers) / n,
        "metrics": merged_metrics,
        "answers": answers,
    }


def detect_gpu_slots(workers_per_gpu: int, idle_util_threshold: int, min_free_mb: int) -> list[int]:
    """Return a list of GPU indices, one entry per worker slot ("anything idle":
    a GPU qualifies if its utilization is under the threshold and it has enough
    free memory for at least one worker)."""
    q = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    slots: list[int] = []
    for line in q.stdout.strip().splitlines():
        idx, used, total, util = [int(x.strip()) for x in line.split(",")]
        free = total - used
        if util > idle_util_threshold:
            print(f"{LOG} GPU {idx}: busy (util {util}% > {idle_util_threshold}%), skipping", flush=True)
            continue
        n_fit = min(workers_per_gpu, free // min_free_mb)
        print(f"{LOG} GPU {idx}: util {util}%, free {free} MiB -> {n_fit} worker slot(s)", flush=True)
        slots.extend([idx] * n_fit)
    return slots


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("csd_path")
    p.add_argument("--dataset", required=True)
    p.add_argument("--sample-size", type=int, required=True)
    p.add_argument("--output-json", required=True)
    # Split names are required (no default) — silently defaulting to one side
    # is how the 2026-07-17 bar/eval split mixup happened.
    p.add_argument("--gsm-split-file")
    p.add_argument("--gsm-split-name", choices=["train", "test"], default=None)
    p.add_argument("--spider-split-file")
    p.add_argument("--spider-split-name", choices=["train", "test"], default=None)
    p.add_argument("--workers-per-gpu", type=int, default=2)
    p.add_argument("--stagger-seconds", type=int, default=45)
    p.add_argument("--gpu-util", type=float, default=0.18,
                   help="vllm gpu_memory_utilization PER WORKER")
    p.add_argument("--idle-util-threshold", type=int, default=30)
    p.add_argument("--min-free-mb", type=int, default=8000)
    args, passthrough = p.parse_known_args()

    split_file = args.spider_split_file if args.dataset == "spider" else args.gsm_split_file
    split_name = args.spider_split_name if args.dataset == "spider" else args.gsm_split_name
    if not split_file:
        print(f"{LOG} ERROR: a split file is required for sharding", flush=True)
        return 2
    if not split_name:
        print(f"{LOG} ERROR: an explicit split name is required when a split file is "
              f"given (no default side — see synthesis/split_provenance.py)", flush=True)
        return 2
    indices_key = indices_key_for(args.dataset, split_name)

    split = json.loads(Path(split_file).read_text())
    n = min(args.sample_size, len(split[indices_key]))

    slots = detect_gpu_slots(args.workers_per_gpu, args.idle_util_threshold, args.min_free_mb)
    if not slots:
        print(f"{LOG} ERROR: no idle GPU capacity found", flush=True)
        return 2
    shards = plan_shards(n, len(slots))
    print(f"{LOG} {n} examples -> {len(shards)} shard(s) on GPU slots {slots[:len(shards)]}", flush=True)

    out_base = Path(args.output_json)
    work_dir = out_base.parent / f"{out_base.stem}_shards"
    work_dir.mkdir(parents=True, exist_ok=True)

    split_flag = "--spider-split-file" if args.dataset == "spider" else "--gsm-split-file"
    name_flag = "--spider-split-name" if args.dataset == "spider" else "--gsm-split-name"

    procs = []
    for i, ((lo, hi), gpu) in enumerate(zip(shards, slots)):
        shard_split_path = work_dir / f"split_shard{i}.json"
        shard_split_path.write_text(json.dumps(slice_split(split, indices_key, lo, hi)))
        part_json = work_dir / f"part{i}.json"
        log_path = work_dir / f"part{i}.log"
        cmd = [
            sys.executable, "-m", "synthesis.scripts.reevaluate_compiled_csd", args.csd_path,
            "--dataset", args.dataset, "--sample-size", str(hi - lo),
            split_flag, str(shard_split_path), name_flag, split_name,
            "--vllm-gpu-memory-utilization", str(args.gpu_util),
            "--output-json", str(part_json), *passthrough,
        ]
        if i > 0:
            print(f"{LOG} stagger {args.stagger_seconds}s before worker {i}", flush=True)
            time.sleep(args.stagger_seconds)
        print(f"{LOG} worker {i}: examples [{lo},{hi}) on GPU {gpu}, log {log_path}", flush=True)
        with open(log_path, "w") as lf:
            procs.append((i, part_json, subprocess.Popen(
                cmd, stdout=lf, stderr=subprocess.STDOUT,
                env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
            )))

    failures = []
    parts = []
    for i, part_json, proc in procs:
        rc = proc.wait()
        if rc != 0 or not part_json.exists():
            failures.append(i)
            print(f"{LOG} worker {i} FAILED (rc={rc}); log: {work_dir}/part{i}.log", flush=True)
        else:
            parts.append(json.loads(part_json.read_text()))
            print(f"{LOG} worker {i} done", flush=True)

    if failures:
        print(f"{LOG} {len(failures)} shard(s) failed: {failures}. Part files kept in {work_dir}; "
              f"NOT writing merged output.", flush=True)
        return 1

    merged = merge_results(parts)
    out_base.write_text(json.dumps(merged, indent=1))
    print(f"{LOG} merged {len(merged['answers'])} examples -> {out_base} "
          f"acc={merged['accuracy']:.4f} syn={merged['syntax_rate']:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
