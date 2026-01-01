#!/usr/bin/env python3
"""
Run GSM-symbolic evaluation using a compiled generated CSD program.

Expected workflow:
  - compile Dafny to Python into GeneratedCSD-py/ (overwrite-in-place)
  - ensure GeneratedCSD-py/module_.py provides extern implementations
  - run this script to produce outputs + summary metrics
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List


def load_gsm_items(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    items: List[Dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        items.append(json.loads(line))
    return items


def gsm_prompt(question: str) -> str:
    return (
        "You are a math reasoning assistant. For the following word problem, return your answer "
        "as a single expression wrapped in double angle brackets << >>.\n\n"
        f"Problem:\n{question}\n\n"
        "Answer:\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compiled-py-dir", default="GeneratedCSD-py")
    ap.add_argument("--dataset-path", default="./datasets/ml-gsm-symbolic/generated_data/GSM_symbolic.jsonl")
    ap.add_argument("--results-dir", default="./outputs/generated-csd/")
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0, help="If >0, limit examples")
    args = ap.parse_args()

    compiled_py_dir = os.path.abspath(args.compiled_py_dir)
    dataset_path = os.path.abspath(args.dataset_path)
    results_dir = os.path.abspath(args.results_dir)
    os.makedirs(results_dir, exist_ok=True)

    sys.path.insert(0, compiled_py_dir)
    import importlib

    module_ = importlib.import_module("module_")
    Generated = importlib.import_module("GeneratedCSD")
    CSD_mod = importlib.import_module("CSD")

    items = load_gsm_items(dataset_path)
    if args.limit and args.limit > 0:
        items = items[: args.limit]

    program = Generated.default__.GeneratedProgram()

    outputs: List[Dict[str, Any]] = []
    agg = {
        "n": 0,
        "parse_ok": 0,
        "unconstrained_calls": 0,
        "constrained_calls": 0,
        "total_latency_s": 0.0,
    }

    for obj in items:
        q = obj.get("question", "")
        prompt = gsm_prompt(q)
        module_.reset_metrics()
        t0 = time.time()
        out = CSD_mod.default__.Run(program, prompt, int(args.max_steps))
        wall = time.time() - t0

        ok = bool(module_.ParseOk(0, out))
        m = dict(module_.METRICS)
        m["wall_time_s"] = wall

        agg["n"] += 1
        agg["parse_ok"] += 1 if ok else 0
        agg["unconstrained_calls"] += int(m.get("unconstrained_calls", 0))
        agg["constrained_calls"] += int(m.get("constrained_calls", 0))
        agg["total_latency_s"] += float(m.get("total_latency_s", 0.0))

        outputs.append(
            {
                "id": obj.get("id"),
                "question": q,
                "output": out,
                "parse_ok": ok,
                "metrics": m,
            }
        )

    with open(os.path.join(results_dir, "outputs.json"), "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2)

    summary = dict(agg)
    summary["parse_ok_rate"] = (agg["parse_ok"] / agg["n"]) if agg["n"] else 0.0
    with open(os.path.join(results_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


