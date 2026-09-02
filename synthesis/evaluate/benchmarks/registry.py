"""Benchmark logic registry for evaluation delegation."""

from __future__ import annotations

from importlib import import_module
from typing import Any


def get_logic(dataset_name: str) -> Any:
    if dataset_name == "gsm_symbolic":
        return import_module("synthesis.evaluate.benchmarks.gsm_symbolic.eval_logic")
    if dataset_name == "spider":
        return import_module("synthesis.evaluate.benchmarks.sql_spider.eval_logic")
    if dataset_name == "smiles":
        return import_module("synthesis.evaluate.benchmarks.smiles.eval_logic")
    raise ValueError(f"Unknown dataset: {dataset_name}")


def resolve_require_delimiters(dataset_name: str, cli_value: bool) -> bool:
    """Decide whether the eval loop should require a visible << >> span.

    A benchmark that never emits << >> (right now: Spider in token-0 mode,
    and SMILES) can't produce that span no matter what the caller asks for --
    asking for it anyway just turns a constant ("no visible delimiters") into
    something that looks like a real failure symptom. So if the benchmark
    says it can't emit delimiters, the answer is always False, overriding the
    CLI flag. Otherwise the CLI flag (`cli_value`) decides, same as before.
    """
    logic = get_logic(dataset_name)
    if not logic.emits_visible_delimiters():
        return False
    return cli_value
