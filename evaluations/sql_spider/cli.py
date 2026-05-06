#!/usr/bin/env python3
"""
SQL Spider evaluation CLI.

Runs either the CSD strategy or the unconstrained baseline on the Spider
dev split, then scores with Spider's execution-accuracy evaluator.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import torch

from evaluations.sql_spider.dataset import (
    default_db_dir,
    default_tables_json,
    load_spider,
)
from evaluations.sql_spider.grammar import (
    parse_db_info,
    build_dynamic_sql_grammar,
    extract_schema_identifiers,
)
from evaluations.sql_spider.metrics import SQLMetrics
from evaluations.sql_spider.generation import run_crane_csd, run_unconstrained
from evaluations.sql_spider.environment import (
    setup_dafny_environment,
    verify_critical_tokens,
)
from evaluations.sql_spider.executor import execute_accuracy, _clean_sql
from evaluations.common.parser_utils import create_lark_dafny_parser

PROJECT_ROOT = Path(__file__).parent.parent.parent


PROMPT_TEMPLATE = (
    "You are given a database schema and a question. "
    "Write a SINGLE SQL query answering the question, using ONLY the tables and columns in the schema. "
    "Only output the SQL query. Stop immediately after the query (no explanation, no code fences).\n\n"
    "Example:\n"
    "db_id: concert_singer\n"
    "db_info: # singer ( singer_id , name , country , age )\n"
    "question: How many singers do we have?\n"
    "SQL: SELECT count(*) FROM singer\n\n"
    "{prompt_body}\n"
    "SQL:"
)


def _extract_sql(output_text: str) -> str:
    """Pull the SQL out of the model output.

    Matches itergen's setup: the model emits a bare SQL query and stops.
    Take everything up to the first blank-line boundary and strip.
    """
    if not output_text:
        return ""
    return _clean_sql(output_text.split("\n\n")[0])


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate Spider text-to-SQL with CSD")
    ap.add_argument("--run-dir", type=Path, required=True,
                    help="Path to compiled CSD run directory")
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct",
                    help="HuggingFace model ID")
    ap.add_argument("--backend", default="huggingface",
                    choices=["huggingface", "vllm"],
                    help="Runtime backend")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--source", default="local", choices=["auto", "hf", "local"],
                    help="Spider data source")
    ap.add_argument("--spider-dir", type=Path, default=None,
                    help="Local Spider directory (dev.json, tables.json, databases/)")
    ap.add_argument("--db-dir", type=Path, default=None,
                    help="SQLite databases directory for execution accuracy")
    ap.add_argument("--tables-json", type=Path, default=None,
                    help="Spider tables.json")
    ap.add_argument("--limit", type=int, default=50,
                    help="Max examples to evaluate")
    ap.add_argument("--split-file", type=Path, default=None,
                    help="Optional Spider train/test split manifest")
    ap.add_argument("--split-name", choices=["train", "test", "eval"], default="test",
                    help="Split key to read from --split-file")
    ap.add_argument("--random-sample", action="store_true",
                    help="Randomly sample examples instead of taking first N")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=400,
                    help="Max generation steps per example")
    ap.add_argument("--grammar", type=Path,
                    default=PROJECT_ROOT / "grammars" / "sql.lark",
                    help="Base SQL grammar")
    ap.add_argument("--no-dynamic-grammar", action="store_true",
                    help="Disable per-example schema constraining")
    ap.add_argument("--etype", default="exec", choices=["exec", "match", "all"],
                    help="Spider evaluator metric type")
    ap.add_argument("--unconstrained", action="store_true",
                    help="Run unconstrained baseline instead of CSD")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--load-in-8bit", action="store_true")
    ap.add_argument("--vllm-tensor-parallel-size", type=int, default=None)
    ap.add_argument("--vllm-pipeline-parallel-size", type=int, default=1)
    ap.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.8)
    ap.add_argument("--vllm-max-model-len", type=int, default=4096)
    ap.add_argument("--vllm-enforce-eager", action="store_true", default=True)
    ap.add_argument("--pred-dump", type=Path, default=None,
                    help="Optional JSON output with predictions, scores, and rows")
    args = ap.parse_args()

    split_indices = None
    if args.split_file is not None:
        manifest = json.loads(args.split_file.read_text())
        key = f"{args.split_name}_indices"
        if key not in manifest:
            available = sorted(k for k in manifest if k.endswith("_indices"))
            raise SystemExit(
                f"{args.split_file} does not contain {key}; available index fields: {available}"
            )
        split_indices = manifest[key]

    examples = load_spider(
        source=args.source,
        spider_dir=args.spider_dir,
        limit=args.limit,
        random_sample=args.random_sample and split_indices is None,
        seed=args.seed,
        indices=split_indices,
    )
    n = len(examples)
    metrics = SQLMetrics()

    print(f"Setting up Dafny environment (backend={args.backend})...")
    grammar_text = args.grammar.read_text()
    parser_factory_cache: dict = {}

    dafny_env = setup_dafny_environment(
        run_dir=args.run_dir,
        model_name=args.model,
        backend=args.backend,
        device=args.device,
        grammar_file=args.grammar,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        vllm_pipeline_parallel_size=args.vllm_pipeline_parallel_size,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_max_model_len=args.vllm_max_model_len,
        vllm_enforce_eager=args.vllm_enforce_eager,
    )
    verify_critical_tokens(dafny_env["tokenizer"])
    print("Model loaded. Starting evaluation...\n")

    predictions: list[str] = []

    for i, example in enumerate(examples):
        prompt_body = example.get("prompt") or (
            f"db_id: {example.get('db_id', '')}\n"
            f"db_info: {example.get('db_info', '')}\n"
            f"question: {example.get('question', '')}"
        )
        prompt = PROMPT_TEMPLATE.format(prompt_body=prompt_body)

        # Build a schema-constrained parser for this question, if enabled.
        dynamic_parser = None
        tables, columns = [], []
        if not args.no_dynamic_grammar and not args.unconstrained:
            tables, columns = extract_schema_identifiers(example.get("db_info", ""))
            cache_key = (tuple(tables), tuple(columns))
            parser_factory = parser_factory_cache.get(cache_key)
            if parser_factory is None:
                dynamic_grammar = build_dynamic_sql_grammar(grammar_text, tables, columns)
                parser_factory = create_lark_dafny_parser(
                    dynamic_grammar,
                    dafny_env["VerifiedDecoderAgent"],
                    dafny_env["_dafny"],
                    start="csd_start",
                    tokenizer=dafny_env["tokenizer"],
                )
                parser_factory_cache[cache_key] = parser_factory
            dynamic_parser = parser_factory(dafny_env["lm"]._Tokens)

        # Build per-example valid_token_groups. The outer list is one group per
        # schema table (in schema order). The first group is the table-name
        # group; subsequent groups are the columns belonging to each table.
        # Tokens within a group are BPE-token strings produced by tokenizing
        # each identifier in common case/space variants.
        valid_token_groups: list = []
        if not args.unconstrained and (tables or columns):
            tok = dafny_env["tokenizer"]

            def _tokens_for(ident: str) -> list:
                bag = set()
                for variant in (ident, " " + ident, ident.lower(), " " + ident.lower(),
                                ident.upper(), " " + ident.upper()):
                    try:
                        ids = tok.encode(variant, add_special_tokens=False)
                    except Exception:
                        continue
                    for tid in ids:
                        try:
                            bag.add(tok.decode([tid]))
                        except Exception:
                            pass
                return sorted(bag)

            schema = parse_db_info(example.get("db_info", ""))
            table_group = []
            for tbl in sorted(schema.keys()):
                table_group.extend(_tokens_for(tbl))
            valid_token_groups.append(sorted(set(table_group)))
            for tbl in sorted(schema.keys()):
                col_bag = []
                for col in schema[tbl]:
                    col_bag.extend(_tokens_for(col))
                valid_token_groups.append(sorted(set(col_bag)))

        print(f"[{i+1}/{n}] {example.get('db_id', '')}: {example.get('question', '')[:80]}", flush=True)

        if args.unconstrained:
            out_text, tok_count, dt = run_unconstrained(
                dafny_env, prompt, args.max_steps, debug=args.verbose
            )
            constrained_segments = []
        else:
            out_text, tok_count, dt, constrained_segments, _ = run_crane_csd(
                dafny_env, prompt, args.max_steps, args.grammar,
                debug_delimiters=args.verbose,
                dynamic_parser=dynamic_parser,
                valid_token_groups=valid_token_groups,
            )

        contains_delimiters = True  # not used for SQL; launch with --no-require-delimiters
        pred_sql = _extract_sql(out_text)
        predictions.append(pred_sql)

        metrics.update_generation(
            contains_delimiters=contains_delimiters,
            token_count=tok_count,
            time_seconds=dt,
            constrained_segments=constrained_segments,
        )

        avg_time = metrics.total_time / max(1, metrics.n)
        eta_s = avg_time * (n - metrics.n)
        eta = f"{eta_s:.0f}s" if eta_s < 60 else f"{eta_s/60:.1f}m"
        print(f"  -> pred: {pred_sql[:120]}", flush=True)
        print(f"  -> Tokens: {tok_count} | Time: {dt:.2f}s | ETA: {eta}", flush=True)

        if args.verbose:
            print(f"  Gold: {example.get('query', '')}")
            print(f"  Output (raw): {out_text[:400]}")

    print("\nScoring predictions with Spider evaluator...")
    scores, error_types, rows = execute_accuracy(
        predictions=predictions,
        examples=examples,
        db_dir=args.db_dir or default_db_dir(),
        tables_json=args.tables_json or default_tables_json(),
        etype=args.etype,
    )
    metrics.record_batch_scores(scores, error_types)

    # Dump full predictions + per-row scoring results for offline diagnosis.
    try:
        import json, os
        dump_path = os.environ.get("SQL_PRED_DUMP")
        if dump_path:
            with open(dump_path, "w") as _f:
                json.dump({
                    "predictions": predictions,
                    "rows": rows,
                    "scores": scores,
                    "error_types": error_types,
                }, _f, indent=2, default=str)
            print(f"Wrote prediction dump to {dump_path}")
    except Exception as _e:
        print(f"Failed to dump predictions: {_e}")

    print("\n" + "=" * 60)
    print("SQL SPIDER RESULTS")
    print("=" * 60)
    print(f"Method: {'Unconstrained Baseline' if args.unconstrained else 'CSD'}")
    print(f"Model: {args.model}")
    print(f"Examples scored: {len(rows)}")
    print()
    print(metrics.summary())

    pred_dump = args.pred_dump
    if pred_dump is None and os.environ.get("SQL_PRED_DUMP"):
        pred_dump = Path(os.environ["SQL_PRED_DUMP"])
    if pred_dump is not None:
        pred_dump.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "method": "Unconstrained Baseline" if args.unconstrained else "CSD",
            "model": args.model,
            "split_file": str(args.split_file) if args.split_file else None,
            "split_name": args.split_name if args.split_file else None,
            "scores": scores,
            "error_types": error_types,
            "metrics_summary": metrics.summary(),
            "rows": rows,
        }
        pred_dump.write_text(json.dumps(payload, indent=2))
        print(f"Wrote prediction dump: {pred_dump}")


if __name__ == "__main__":
    main()
