"""One-off targeted experiment: time a fixed compiled SMILES CSD module under
two settings (control: ms=750, treatment: ms=200), using the same Evaluator
codepath that synthesis feedback / final benchmark uses.

Purpose: validate that the recent changes (#1 lower max-steps, #2 stop-at-100-
unique-valid hook) produce the expected wall-time and per-sample distribution
changes on the SMILES acrylates class. Disposable.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path("/home/aadivyar/csd-generation")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from synthesis.evaluate.evaluator import Evaluator


def percentile(sorted_xs, q):
    if not sorted_xs:
        return None
    k = max(0, min(len(sorted_xs) - 1, int(round(q * (len(sorted_xs) - 1)))))
    return sorted_xs[k]


def run_one(label, compiled_module_path, sample_size, max_steps, smiles_class):
    print(f"=== {label}: ms={max_steps} n={sample_size} class={smiles_class} ===", flush=True)
    evaluator = Evaluator(
        dataset_name="smiles",
        model_name="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        backend="vllm",
        device="cuda",
        sample_size=sample_size,
        max_steps=max_steps,
        step_token_budget=1,
        vllm_gpu_memory_utilization=0.30,
        vllm_max_model_len=4096,
        max_seconds_per_example=None,
        smiles_classes=smiles_class,
    )
    t0 = time.time()
    result = evaluator.evaluate_sample(compiled_module_path=Path(compiled_module_path))
    wall = time.time() - t0

    so = result.sample_outputs or []
    ts = sorted(float(s.get("time_seconds") or 0.0) for s in so)
    toks = [int(s.get("token_count") or 0) for s in so]
    summary = {
        "label": label,
        "max_steps": max_steps,
        "requested_sample_size": sample_size,
        "actually_evaluated": len(so),
        "success": bool(result.success),
        "error": result.error,
        "early_stopped": bool(result.early_stopped),
        "early_stop_reason": result.early_stop_reason,
        "total_wall_s": round(wall, 2),
        "per_sample_total_s": round(sum(ts), 2),
        "per_sample_mean_s": round(statistics.mean(ts), 3) if ts else None,
        "per_sample_p50_s": round(percentile(ts, 0.50), 3) if ts else None,
        "per_sample_p90_s": round(percentile(ts, 0.90), 3) if ts else None,
        "per_sample_max_s": round(ts[-1], 3) if ts else None,
        "tokens_mean": round(statistics.mean(toks), 1) if toks else None,
        "tokens_max": max(toks) if toks else None,
        "tokens_hit_budget": sum(1 for t in toks if t >= max_steps),
        "syntax_rate": round(result.syntax_rate, 3),
        "accuracy": round(result.accuracy, 3),
        "smiles_paper_trial": (result.aux_metrics or {}).get("smiles_paper_trial"),
    }
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiled", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--class-name", default="acrylates")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    runs = []
    runs.append(run_one("A_control_ms750_n50", args.compiled, 50, 750, args.class_name))
    runs.append(run_one("C_treatment_ms400_n50", args.compiled, 50, 400, args.class_name))

    out_path.write_text(json.dumps({"runs": runs}, indent=2))
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
