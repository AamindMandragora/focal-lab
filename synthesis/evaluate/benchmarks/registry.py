"""Benchmark logic registry for evaluation delegation."""

from __future__ import annotations

from importlib import import_module
from typing import Any


def get_logic(dataset_name: str) -> Any:
    if dataset_name == "gsm_symbolic":
        return import_module("synthesis.evaluate.benchmarks.gsm_symbolic.eval_logic")
    if dataset_name == "spider":
        return import_module("synthesis.evaluate.benchmarks.sql_spider.eval_logic")
    raise ValueError(f"Unknown dataset: {dataset_name}")
