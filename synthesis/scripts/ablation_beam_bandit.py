#!/usr/bin/env python3
"""
Ablation: refinement beam size x helper-selection policy (utility vs bandit).

Runs `python -m synthesis.run_synthesis` for each grid cell, then reads
`outputs/generated/latest/results/{success,failure}_report.json` for metrics.

Example (local vLLM, small eval for a quick grid):

  PYTHONPATH=. CUDA_VISIBLE_DEVICES=2,3 python synthesis/scripts/ablation_beam_bandit.py \\
    --eval-sample-size 8 --max-iterations 3

OpenAI generation (requires OPENAI_API_KEY / synthesis/.env):

  PYTHONPATH=. CUDA_VISIBLE_DEVICES=2,3 python synthesis/scripts/ablation_beam_bandit.py \\
    --generation-backend openai --generation-model gpt-5.4 \\
    --eval-sample-size 15 --max-iterations 5 --min-accuracy 0.4 --min-syntax-rate 0.8
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _parse_report(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, Any] = {
        "report": path.name,
        "total_attempts": data.get("total_attempts"),
        "task_description": data.get("task_description"),
    }
    # success_report.json (top-level)
    ev = data.get("evaluation_result")
    if isinstance(ev, dict):
        out["accuracy"] = ev.get("accuracy")
        out["syntax_rate"] = ev.get("syntax_rate")
        out["num_correct"] = ev.get("num_correct")
        out["num_examples"] = ev.get("num_examples")
        out["contains_delimiters"] = ev.get("contains_delimiters")
        out["synthesis_success"] = bool(ev.get("success"))
        return out
    # failure_report.json: scan attempts for last evaluation block
    best_acc = None
    best_syn = None
    best_att = None
    for att in data.get("attempts") or []:
        ev2 = att.get("evaluation")
        if not isinstance(ev2, dict):
            continue
        acc = ev2.get("accuracy")
        syn = ev2.get("syntax_rate")
        if isinstance(acc, (int, float)):
            if best_acc is None or acc > best_acc:
                best_acc = acc
                best_syn = syn
                best_att = att.get("attempt_number")
    out["best_eval_attempt"] = best_att
    out["accuracy"] = best_acc
    out["syntax_rate"] = best_syn
    out["synthesis_success"] = False
    return out


def _collect_latest_metrics(repo: Path) -> dict[str, Any]:
    latest = (repo / "outputs" / "generated" / "latest").resolve()
    results_dir = latest / "results"
    for name in ("success_report.json", "failure_report.json"):
        p = results_dir / name
        if p.exists():
            m = _parse_report(p)
            m["run_dir"] = str(latest)
            return m
    return {"run_dir": str(latest), "error": "no success_report or failure_report"}


def main() -> None:
    repo = _repo_root()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--beams", default="1,2,4", help="Comma-separated refinement beam sizes")
    p.add_argument(
        "--policies",
        default="utility,bandit",
        help="Comma-separated helper-selection-policy values (utility|bandit)",
    )
    p.add_argument(
        "--task",
        default="Solve math word problems with constrained symbolic expressions.",
    )
    p.add_argument("--dataset", default="gsm_symbolic")
    p.add_argument("--generation-backend", default="vllm")
    p.add_argument("--generation-model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    p.add_argument("--eval-backend", default="vllm")
    p.add_argument("--eval-model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    p.add_argument("--eval-sample-size", type=int, default=8)
    p.add_argument("--eval-max-steps", type=int, default=512)
    p.add_argument("--max-iterations", type=int, default=3)
    p.add_argument("--min-accuracy", type=float, default=0.0)
    p.add_argument("--min-syntax-rate", type=float, default=0.0)
    p.add_argument(
        "--vllm-max-model-len",
        type=int,
        default=32768,
        help="vLLM context (generation and eval); GSM synthesis prompts often need 32k+",
    )
    p.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=0.8,
        help="Passed to run_synthesis (--vllm-gpu-memory-utilization)",
    )
    p.add_argument(
        "--vllm-tensor-parallel-size",
        type=int,
        default=None,
        help="If set, passed to run_synthesis (use 1 with one visible GPU for long context)",
    )
    p.add_argument(
        "--no-adaptive-helper-mask",
        action="store_true",
        help="Disable adaptive helper mask (not recommended for bandit ablation)",
    )
    p.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Write combined results JSON (default: outputs/ablations/beam_bandit_<ts>.json)",
    )
    p.add_argument("--dry-run", action="store_true", help="Print commands only")
    args = p.parse_args()

    beams = [int(x.strip()) for x in args.beams.split(",") if x.strip()]
    policies = [x.strip() for x in args.policies.split(",") if x.strip()]
    for pol in policies:
        if pol not in {"utility", "bandit"}:
            raise SystemExit(f"Invalid policy {pol!r}; use utility or bandit")

    if args.output_json is None:
        out_dir = repo / "outputs" / "ablations"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.output_json = out_dir / f"beam_bandit_{ts}.json"

    env = os.environ.copy()
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    # Ensure repo root is importable for `python -m synthesis.run_synthesis`
    _pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(repo) if not _pp else f"{repo}{os.pathsep}{_pp}"

    rows: list[dict[str, Any]] = []
    for beam in beams:
        for policy in policies:
            name = f"ablat_b{beam}_{policy}"
            cmd = [
                sys.executable,
                "-m",
                "synthesis.run_synthesis",
                "--task",
                args.task,
                "--dataset",
                args.dataset,
                "--generation-backend",
                args.generation_backend,
                "--generation-model",
                args.generation_model,
                "--eval-backend",
                args.eval_backend,
                "--eval-model",
                args.eval_model,
                "--min-accuracy",
                str(args.min_accuracy),
                "--min-syntax-rate",
                str(args.min_syntax_rate),
                "--max-iterations",
                str(args.max_iterations),
                "--eval-sample-size",
                str(args.eval_sample_size),
                "--eval-max-steps",
                str(args.eval_max_steps),
                "--refinement-beam-size",
                str(beam),
                "--helper-selection-policy",
                policy,
                "--vllm-max-model-len",
                str(args.vllm_max_model_len),
                "--vllm-gpu-memory-utilization",
                str(args.vllm_gpu_memory_utilization),
                "--output-name",
                name,
            ]
            if args.vllm_tensor_parallel_size is not None:
                cmd.extend(
                    ["--vllm-tensor-parallel-size", str(args.vllm_tensor_parallel_size)]
                )
            if args.no_adaptive_helper_mask:
                cmd.append("--no-adaptive-helper-mask")

            row: dict[str, Any] = {
                "refinement_beam_size": beam,
                "helper_selection_policy": policy,
                "cmd": cmd,
            }
            if args.dry_run:
                rows.append(row)
                continue

            t0 = time.time()
            proc = subprocess.run(cmd, cwd=repo, env=env)
            row["exit_code"] = proc.returncode
            row["wall_seconds"] = round(time.time() - t0, 3)
            row.update(_collect_latest_metrics(repo))
            rows.append(row)
            print(
                f"[ablation] beam={beam} policy={policy} exit={proc.returncode} "
                f"acc={row.get('accuracy')} syntax={row.get('syntax_rate')} "
                f"dir={row.get('run_dir')}",
                flush=True,
            )

    cfg = vars(args).copy()
    if isinstance(cfg.get("output_json"), Path):
        cfg["output_json"] = str(cfg["output_json"])
    payload = {
        "config": cfg,
        "results": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
