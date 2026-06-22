"""Adapter that subprocesses the upstream CRANE repo's main.py for GSM/Spider baselines.

Provides paper-faithful evaluation for the unconstrained, gcd, crane, and itergen
strategies on GSM-Symbolic and Spider, using the published CRANE evaluation
pipeline (8-shot, greedy decoding, the paper's prompt template, and mode flags
per strategy).

Strategy -> CRANE main.py flag mapping:

    unconstrained: --cot_grammar_mode original --do_cot True
    gcd:           --cot_grammar_mode itergen  --do_cot False
    crane:         --cot_grammar_mode adaptive --do_cot True
    itergen:       --cot_grammar_mode itergen  --do_cot True

The CRANE repo lives at ``~/CRANE`` by default; override via ``CRANE_REPO_DIR``.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


CRANE_REPO_DIR = Path(
    os.environ.get("CRANE_REPO_DIR", os.path.expanduser("~/CRANE"))
).resolve()
CRANE_SRC_DIR = CRANE_REPO_DIR / "src"


_STRATEGY_TO_CRANE_MODE: dict[str, tuple[str, bool]] = {
    "unconstrained": ("original", True),
    "gcd": ("itergen", False),
    "crane": ("adaptive", True),
    "itergen": ("itergen", True),
}


_DATASET_TO_CRANE_GRAMMAR: dict[str, str] = {
    "gsm_symbolic": "gsm",
    "spider": "sql",
}


def _latest_crane_results(crane_src_dir: Path, dataset: str) -> Path | None:
    dataset_dir = crane_src_dir / "logging" / dataset
    if not dataset_dir.exists():
        return None
    candidates = sorted(
        dataset_dir.rglob("*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def run_crane_repo_baseline(args: argparse.Namespace, dataset: str) -> int:
    """Subprocess the upstream CRANE main.py and write our standard baseline JSON."""
    from synthesis.evaluate.run_legacy_fixed_strategy import (
        _annotate_legacy_rows_with_syntax,
        _build_minimal_json,
        _legacy_local_cuda_device,
    )

    if dataset not in _DATASET_TO_CRANE_GRAMMAR:
        raise ValueError(
            f"crane_repo_runner supports gsm_symbolic and spider only; got {dataset}"
        )
    if args.strategy not in _STRATEGY_TO_CRANE_MODE:
        raise ValueError(
            f"crane_repo_runner supports unconstrained/gcd/crane/itergen; got {args.strategy}"
        )
    if not CRANE_SRC_DIR.exists():
        raise RuntimeError(
            f"Upstream CRANE source directory not found: {CRANE_SRC_DIR}. "
            "Set CRANE_REPO_DIR to override."
        )

    mode, do_cot = _STRATEGY_TO_CRANE_MODE[args.strategy]
    grammar = _DATASET_TO_CRANE_GRAMMAR[dataset]
    is_grammar_constrained = mode != "original"
    grammar_flag = grammar if is_grammar_constrained else "text"

    # Controlled-comparison: when a GSM split manifest is given, evaluate EXACTLY the
    # split's eval examples (the same set CARS/metaDecode use) by passing them to
    # CRANE main.py's --indices flag. CRANE's loader is sorted (utils.py), so these
    # positions index into sorted(glob('*.json')) just like our split manifests do.
    gsm_split_indices = None
    if dataset == "gsm_symbolic":
        split_file = getattr(args, "gsm_split_file", None)
        if split_file:
            split_name = getattr(args, "gsm_split_name", "eval")
            with open(split_file) as f:
                manifest = json.load(f)
            key = f"{split_name}_indices"
            if key not in manifest:
                available = sorted(k for k in manifest if k.endswith("_indices"))
                raise ValueError(
                    f"Split file {split_file} does not contain {key}. "
                    f"Available index fields: {available}"
                )
            gsm_split_indices = manifest[key]
            if not isinstance(gsm_split_indices, list) or not all(
                isinstance(i, int) for i in gsm_split_indices
            ):
                raise ValueError(f"{key} in {split_file} must be a list of integers")

    # With explicit indices, the example count is fixed by the split, not --eval-sample-size.
    num_examples_arg = (
        str(len(gsm_split_indices))
        if gsm_split_indices is not None
        else str(args.eval_sample_size)
    )

    cmd = [
        "python",
        "main.py",
        "--dataset", dataset,
        "--num_examples", num_examples_arg,
        "--num_shots", "8",
        "--overwrite_results", "True",
        "--write_file", "True",
        "--regex_parser", "True",
        "--modify_system_prompt", "True",
        "--cot_model", args.eval_model,
        "--cot_grammar_mode", mode,
        "--cot_grammar", grammar_flag,
        "--out_grammar", grammar_flag,
        "--max_tokens", str(args.eval_max_steps),
        "--temperature", "0.0",
        "--start_symbol", "<<",
        "--end_symbol", ">>",
    ]
    if do_cot:
        cmd.extend(["--do_cot", "True"])
    if gsm_split_indices is not None:
        cmd.extend(["--indices", ",".join(str(i) for i in gsm_split_indices)])

    legacy_device = _legacy_local_cuda_device(args.device)
    cmd.extend(["--cot_device", legacy_device, "--llm_parser_device", legacy_device])

    env = os.environ.copy()
    extra_pythonpath = []
    syncode_dir = CRANE_REPO_DIR / "syncode"
    if syncode_dir.exists():
        extra_pythonpath.append(str(syncode_dir))
        if (syncode_dir / "syncode").exists():
            extra_pythonpath.append(str(syncode_dir / "syncode"))
    if extra_pythonpath:
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join(
            extra_pythonpath + ([existing] if existing else [])
        )

    started = time.perf_counter()
    subprocess.run(cmd, cwd=str(CRANE_SRC_DIR), check=True, env=env)

    latest = _latest_crane_results(CRANE_SRC_DIR, dataset)
    if latest is None:
        raise RuntimeError(
            f"No CRANE result JSONL found under {CRANE_SRC_DIR}/logging/{dataset}/"
        )

    rows = _load_jsonl(latest)
    rows = _annotate_legacy_rows_with_syntax(rows, args, dataset)

    _build_minimal_json(
        rows,
        args.output_json,
        run_wall_time_seconds=time.perf_counter() - started,
        extra_metrics={
            "adapter": "crane_repo",
            "crane_repo_path": str(CRANE_REPO_DIR),
            "crane_grammar_mode": mode,
            "crane_do_cot": do_cot,
            "crane_num_shots": 8,
            "strategy_mapped": args.strategy,
            "result_file": str(latest),
        },
    )
    print(f"Saved baseline JSON: {args.output_json}")
    return 0
