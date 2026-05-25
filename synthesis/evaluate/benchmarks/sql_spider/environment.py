"""
Environment setup for SQL Spider.

Thin re-export of evaluations.gsm_symbolic.environment.setup_dafny_environment —
the setup is dataset-agnostic, it just reads the grammar file the caller
passes in. Kept as a module so synthesis.evaluate.evaluator can import it via
`synthesis.evaluate.benchmarks.sql_spider.environment`.
"""

from __future__ import annotations

from synthesis.evaluate.benchmarks.gsm_symbolic.environment import (
    load_compiled_modules,
    resolve_run_dir,
    setup_dafny_environment,
    verify_critical_tokens,
)

__all__ = [
    "load_compiled_modules",
    "resolve_run_dir",
    "setup_dafny_environment",
    "verify_critical_tokens",
]
