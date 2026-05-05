#!/usr/bin/env python3
"""Run original IterGen on the SMILES class-generation benchmark."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluations.smiles.dataset import SMILES_CLASSES, get_smiles_task
from evaluations.smiles.metrics import evaluate_smiles_output


def add_itergen_paths(itergen_repo: Path) -> None:
    for path in [
        itergen_repo,
        itergen_repo / "itergen" / "syncode" / "syncode",
    ]:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def normalize_classes(raw: str) -> list[str]:
    classes = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = sorted(set(classes) - set(SMILES_CLASSES))
    if unknown:
        raise ValueError(f"Unknown SMILES class(es): {unknown}")
    return classes


def format_prompt(prompt: str, model: str) -> str | list[dict[str, str]]:
    if any(tag in model for tag in ("Instruct", "instruct", "chat", "it")):
        return [{"role": "user", "content": prompt}]
    return prompt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--itergen-repo", type=Path, default=Path("/home/aadivyar/itergen"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--classes", default=",".join(SMILES_CLASSES))
    parser.add_argument("--target-samples", type=int, default=100)
    parser.add_argument("--max-attempts", type=int, default=2000)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--recurrence-penalty", type=float, default=0.3)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    import torch

    torch.manual_seed(args.seed)
    itergen_repo = args.itergen_repo.expanduser().resolve()
    add_itergen_paths(itergen_repo)
    from itergen.main import IterGen  # type: ignore

    classes = normalize_classes(args.classes)
    do_sample = args.temperature is not None
    class_summaries: list[dict[str, Any]] = []
    start = time.time()
    old_cwd = Path.cwd()
    os.chdir(itergen_repo)
    try:
        for class_name in classes:
            task = get_smiles_task(class_name)
            print(f"[itergen-smiles:{class_name}] loading grammar/model", flush=True)
            iter_gen = IterGen(
                grammar=task["grammar_text"],
                model_id=args.model,
                parse_output_only=True,
                do_sample=do_sample,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                recurrence_penalty=args.recurrence_penalty,
                device=args.device,
            )
            prompt = format_prompt(task["prompt"], args.model)
            rows: list[dict[str, Any]] = []
            unique_valid: set[str] = set()
            total_tokens = 0
            class_start = time.time()
            for attempt in range(1, args.max_attempts + 1):
                if len(unique_valid) >= args.target_samples:
                    break
                raw_output = ""
                metadata: dict[str, Any] = {}
                try:
                    iter_gen.start(prompt)
                    out = iter_gen.forward()
                    raw_output = out[0] if out else ""
                    metadata = dict(iter_gen._metadata or {})
                except Exception as exc:
                    metadata = {"error": str(exc)}
                row_eval = evaluate_smiles_output(
                    class_name,
                    raw_output,
                    task["grammar_text"],
                    task["prompt_exemplars"],
                )
                if row_eval.get("unique_valid_candidate"):
                    unique_valid.add(row_eval.get("smiles", ""))
                token_count = int(metadata.get("total_tokens") or 0)
                total_tokens += token_count
                rows.append({
                    "attempt": attempt,
                    "raw_output": raw_output,
                    "token_count": token_count,
                    "metadata": metadata,
                    **row_eval,
                })
                if attempt % 10 == 0 or len(unique_valid) >= args.target_samples:
                    print(
                        f"  attempts={attempt} unique_valid={len(unique_valid)}/{args.target_samples}",
                        flush=True,
                    )

            syntax_count = sum(1 for row in rows if row.get("syntax_valid"))
            valid_membership_count = sum(1 for row in rows if row.get("valid_class_membership"))
            membership_count_all = sum(1 for row in rows if row.get("class_membership"))
            class_summaries.append({
                "class_name": class_name,
                "attempt_count": len(rows),
                "target_samples": args.target_samples,
                "unique_valid_count": len(unique_valid),
                "reached_target": len(unique_valid) >= args.target_samples,
                "syntax_rate": syntax_count / max(1, len(rows)),
                "accuracy": valid_membership_count / syntax_count if syntax_count else None,
                "accuracy_definition": "class_membership_among_syntax_valid_molecules",
                "accuracy_num_correct": valid_membership_count,
                "accuracy_denominator": syntax_count,
                "invalid_outputs_excluded_from_accuracy": len(rows) - syntax_count,
                "membership_rate_all_attempts": membership_count_all / max(1, len(rows)),
                "avg_num_tokens": total_tokens / max(1, len(rows)),
                "wall_time_seconds": time.time() - class_start,
                "records": rows,
            })
    finally:
        os.chdir(old_cwd)

    output = {
        "config": {
            "method": "itergen",
            "dataset": "smiles",
            "itergen_repo": str(itergen_repo),
            "classes": classes,
            "model": args.model,
            "device": args.device,
            "seed": args.seed,
            "temperature": args.temperature,
            "recurrence_penalty": args.recurrence_penalty,
            "target_samples": args.target_samples,
            "max_attempts": args.max_attempts,
        },
        "classes": class_summaries,
        "wall_time_seconds": time.time() - start,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, default=str))
    print(f"[summary] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
