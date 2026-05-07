#!/usr/bin/env python3
"""Run original IterGen SQL on an explicit Spider split and score with Spider exec accuracy."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.native_libs import ensure_env_lib_first

ensure_env_lib_first()

from evaluations.sql_spider.dataset import (
    default_db_dir,
    default_tables_json,
    load_spider,
)
from evaluations.sql_spider.executor import _clean_sql, execute_accuracy
from project_defaults import default_itergen_repo


def patch_itergen_logits_warper_compat(itergen_cls: type[Any]) -> None:
    if getattr(itergen_cls, "__vas_logits_warper_patched__", False):
        return

    def _compat_update_gen_args(self: Any, **gen_args: dict) -> None:
        from transformers.generation.logits_process import LogitsProcessorList

        self.generation_config.update(**gen_args)
        default_int_fields = {
            "num_return_sequences": 1,
            "num_beams": 1,
            "num_beam_groups": 1,
        }
        for field_name, default_value in default_int_fields.items():
            if getattr(self.generation_config, field_name, None) is None:
                setattr(self.generation_config, field_name, default_value)
        if getattr(self.generation_config, "do_sample", None) is None:
            self.generation_config.do_sample = False
        if hasattr(self.model, "_get_logits_warper"):
            self.logit_warper = self.model._get_logits_warper(self.generation_config, device=self.device)
        else:
            self.logit_warper = LogitsProcessorList()

    itergen_cls.update_gen_args = _compat_update_gen_args
    setattr(itergen_cls, "__vas_logits_warper_patched__", True)


def _load_indices(split_file: Path, split_name: str, limit: int | None) -> list[int]:
    manifest = json.loads(split_file.read_text())
    key = f"{split_name}_indices"
    if key not in manifest:
        available = sorted(k for k in manifest if k.endswith("_indices"))
        raise ValueError(f"{split_file} does not contain {key}; available: {available}")
    indices = list(manifest[key])
    if limit is not None and limit > 0:
        indices = indices[:limit]
    return indices


def _load_itergen_eval_module(itergen_repo: Path):
    eval_path = itergen_repo / "case_studies" / "sql" / "eval_sql.py"
    if not eval_path.exists():
        raise FileNotFoundError(f"Missing IterGen SQL evaluator: {eval_path}")
    import_paths = [
        itergen_repo,
        itergen_repo / "itergen" / "syncode" / "syncode",
    ]
    for import_path in import_paths:
        import_path_str = str(import_path)
        if import_path_str not in sys.path:
            sys.path.insert(0, import_path_str)
    old_cwd = Path.cwd()
    os.chdir(itergen_repo)
    try:
        spec = importlib.util.spec_from_file_location("itergen_sql_eval", eval_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not import {eval_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        os.chdir(old_cwd)


def _question_match(a: str, b: str) -> bool:
    return " ".join(str(a).split()).lower() == " ".join(str(b).split()).lower()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--itergen-repo", type=Path, default=default_itergen_repo())
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--split-name", choices=["train", "test", "eval"], default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-14B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--recurrence-penalty", type=float, default=0.3)
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--db-dir", type=Path, default=None)
    parser.add_argument("--tables-json", type=Path, default=None)
    parser.add_argument("--source", choices=["auto", "hf", "local"], default="auto")
    parser.add_argument("--spider-dir", type=Path, default=None)
    parser.add_argument("--etype", choices=["exec", "match", "all"], default="exec")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    indices = _load_indices(args.split_file, args.split_name, args.limit)
    itergen_repo = args.itergen_repo.expanduser().resolve()
    if not itergen_repo.exists():
        raise FileNotFoundError(f"IterGen repo not found: {itergen_repo}")

    import torch

    torch.manual_seed(args.seed)
    eval_sql = _load_itergen_eval_module(itergen_repo)
    IterGen = eval_sql.IterGen
    patch_itergen_logits_warper_compat(IterGen)
    sql_dataset = eval_sql.sql_dataset

    csd_examples = load_spider(
        source=args.source,
        spider_dir=args.spider_dir,
        indices=indices,
    )
    problems = csd_examples
    mismatches = []
    for local_i, (problem, example, source_idx) in enumerate(zip(problems, csd_examples, indices)):
        if not _question_match(problem.get("question", ""), example.get("question", "")):
            mismatches.append({
                "local_index": local_i,
                "source_index": source_idx,
                "itergen_question": problem.get("question", ""),
                "csd_question": example.get("question", ""),
            })

    do_sample = args.temperature is not None
    gen_kwargs: dict[str, Any] = {
        "do_sample": do_sample,
        "stop_strings": ["\n\n"],
        "max_new_tokens": 200,
        "recurrence_penalty": args.recurrence_penalty,
        "device": args.device,
    }
    if args.temperature is not None:
        gen_kwargs["temperature"] = args.temperature
    iter_gen = IterGen(
        grammar="sql",
        model_id=args.model,
        parse_output_only=True,
        **gen_kwargs,
    )

    predictions: list[str] = []
    rows: list[dict[str, Any]] = []
    start = time.time()
    total_tokens = 0
    for local_i, (source_idx, problem) in enumerate(zip(indices, problems), start=1):
        print(
            f"[{local_i}/{len(problems)}] source_index={source_idx} "
            f"{problem.get('db_id', '')}: {str(problem.get('question', ''))[:80]}",
            flush=True,
        )
        ex_start = time.time()
        out, metadata = eval_sql.generate_sql_with_itergen(
            iter_gen,
            problem,
            max_iter=args.max_iter,
        )
        raw_completion = out[0] if out else ""
        completion = sql_dataset.post_process_answer(raw_completion)
        completion = _clean_sql(completion)
        token_count = int(metadata.get("total_tokens") or 0)
        total_tokens += token_count
        predictions.append(completion)
        rows.append({
            "source_index": source_idx,
            "db_id": problem.get("db_id", ""),
            "question": problem.get("question", ""),
            "prediction": completion,
            "raw_output": raw_completion,
            "token_count": token_count,
            "time_seconds": time.time() - ex_start,
            "metadata": metadata,
        })
        print(f"  -> pred: {completion[:120]}", flush=True)

    print("\nScoring IterGen predictions with Spider evaluator...")
    scores, error_types, scored_rows = execute_accuracy(
        predictions=predictions,
        examples=csd_examples,
        db_dir=args.db_dir or default_db_dir(),
        tables_json=args.tables_json or default_tables_json(),
        etype=args.etype,
    )
    for row, scored in zip(rows, scored_rows):
        validity = scored.get("validity")
        is_correct = scored.get("exec") is True
        row.update({
            "gold": scored.get("gold", ""),
            "exec": scored.get("exec"),
            "validity": validity,
            "is_correct": is_correct,
            "syntax_valid": validity == "Valid",
        })
    num_examples = len(rows)
    num_correct = sum(1 for row in rows if row.get("exec") is True)
    syntax_count = sum(1 for row in rows if row.get("validity") == "Valid")
    all_exec_accuracy = float(scores.get("all", {}).get("exec", 0.0) or 0.0)

    output = {
        "config": {
            "method": "itergen",
            "dataset": "spider",
            "itergen_repo": str(itergen_repo),
            "split_file": str(args.split_file),
            "split_name": args.split_name,
            "indices": indices,
            "model": args.model,
            "device": args.device,
            "seed": args.seed,
            "recurrence_penalty": args.recurrence_penalty,
            "max_iter": args.max_iter,
            "temperature": args.temperature,
        },
        "scores": scores,
        "error_types": error_types,
        "all_exec_accuracy": all_exec_accuracy,
        "accuracy": all_exec_accuracy,
        "num_correct": num_correct,
        "num_examples": num_examples,
        "accuracy_denominator": num_examples,
        "accuracy_definition": "Spider official execution accuracy",
        "syntax_rate": syntax_count / max(1, num_examples),
        "syntax_definition": "Spider executor validity over generated SQL predictions",
        "avg_num_tokens": total_tokens / max(1, num_examples),
        "wall_time_seconds": time.time() - start,
        "question_mismatches": mismatches,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, default=str))
    print(json.dumps({
        "split_name": args.split_name,
        "num_examples": len(rows),
        "all_exec_accuracy": output["all_exec_accuracy"],
        "error_types": error_types,
        "question_mismatches": len(mismatches),
    }, indent=2))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
