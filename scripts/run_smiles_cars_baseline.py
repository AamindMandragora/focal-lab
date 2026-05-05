#!/usr/bin/env python3
"""Run native CARS baseline for SMILES tasks."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluations.smiles.dataset import SMILES_CLASSES
from evaluations.smiles.cars_baseline import run_native_smiles_baseline


def _normalize_classes(raw: str | None) -> list[str]:
    if not raw:
        return list(SMILES_CLASSES)
    classes = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = sorted(set(classes) - set(SMILES_CLASSES))
    if unknown:
        raise ValueError(f"Unknown SMILES class(es): {unknown}")
    return classes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--compiled-module", type=Path, required=True, help="Path to compiled GeneratedCSD.py (used for env bootstrap)")
    ap.add_argument("--classes", type=str, default=",".join(SMILES_CLASSES))
    ap.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--backend", choices=["huggingface", "vllm"], default="vllm")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--max-steps", type=int, default=512)
    ap.add_argument("--target-samples", type=int, default=100)
    ap.add_argument("--max-attempts", type=int, default=1000)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    classes = _normalize_classes(args.classes)
    started = time.time()
    results = run_native_smiles_baseline(
        compiled_module=str(args.compiled_module.expanduser().resolve()),
        classes=classes,
        model_name=args.model_name,
        backend=args.backend,
        device=args.device,
        max_steps=args.max_steps,
        target_samples=args.target_samples,
        max_attempts=args.max_attempts,
        style="cars",
    )
    payload = {
        "method": "cars",
        "config": {
            "compiled_module": str(args.compiled_module),
            "classes": classes,
            "model_name": args.model_name,
            "backend": args.backend,
            "device": args.device,
            "max_steps": args.max_steps,
            "target_samples": args.target_samples,
            "max_attempts": args.max_attempts,
        },
        "wall_time": time.time() - started,
        "results": results,
    }

    output_path = args.output
    if output_path is None:
        output_dir = Path("outputs/smiles-benchmark")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"smiles_cars_native_{int(time.time())}.json"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["config"], indent=2))
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
