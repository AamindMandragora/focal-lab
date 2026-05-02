#!/usr/bin/env python3
"""Prewarm DFA mask stores for GSM-Symbolic dynamic grammars."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluations.common.model_utils import load_runtime_tokenizer
from evaluations.common.parser_utils import prewarm_dfa_mask_store
from evaluations.gsm_symbolic.dataset import load_gsm_symbolic
from evaluations.gsm_symbolic.grammar import build_dynamic_grammar, extract_variables_from_mapping



def main() -> None:
    ap = argparse.ArgumentParser(description="Prewarm GSM DFA mask stores for a fixed sample")
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct", help="Tokenizer/model name")
    ap.add_argument("--backend", choices=["huggingface", "vllm"], default="vllm", help="Tokenizer backend")
    ap.add_argument("--config", choices=["main", "p1", "p2"], default="main")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=10, help="Number of examples to inspect")
    ap.add_argument("--random-sample", action="store_true", help="Randomly sample examples before deduping variable sets")
    ap.add_argument("--seed", type=int, default=None, help="Optional RNG seed for reproducible sampling")
    ap.add_argument("--grammar", type=Path, default=PROJECT_ROOT / "grammars" / "gsm.lark")
    args = ap.parse_args()

    tokenizer = load_runtime_tokenizer(args.model, backend=args.backend)
    dataset = load_gsm_symbolic(
        config=args.config,
        split=args.split,
        limit=args.limit,
        random_sample=args.random_sample,
        seed=args.seed,
    )
    grammar_text = args.grammar.read_text()

    unique_variable_sets = []
    seen = set()
    for example in dataset:
        variable_types = example.get("variable_types") or {}
        if not isinstance(variable_types, dict):
            continue
        variables = tuple(sorted(extract_variables_from_mapping(variable_types)))
        if not variables or variables in seen:
            continue
        seen.add(variables)
        unique_variable_sets.append(variables)

    print(f"Found {len(unique_variable_sets)} unique GSM variable sets across {len(dataset)} examples")

    for idx, variable_set in enumerate(unique_variable_sets, start=1):
        dynamic_grammar = build_dynamic_grammar(grammar_text, list(variable_set))
        preview = list(variable_set[:6])
        suffix = '...' if len(variable_set) > 6 else ''
        print(f"[{idx}/{len(unique_variable_sets)}] Prewarming csd_start DFA for vars={preview}{suffix}")
        prewarm_dfa_mask_store(dynamic_grammar, start="csd_start", tokenizer=tokenizer)

    print("Prewarm complete.")


if __name__ == "__main__":
    main()
