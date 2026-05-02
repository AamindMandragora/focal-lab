"""
SQL Spider evaluation module.

Mirrors evaluations/gsm_symbolic: loads the Spider text-to-SQL benchmark,
builds a schema-constrained grammar, runs CSD generation, and scores with
execution accuracy using the vendored Spider evaluator.

Modules:
- dataset: Spider dataset loading
- grammar: Per-schema dynamic grammar construction
- generation: CSD generation method (wrapper over gsm_symbolic's runner)
- environment: Dafny environment setup (SQL grammar)
- executor: Execution-accuracy scoring via Spider's evaluator
- metrics: Evaluation metrics per hardness
- cli: Command-line interface
"""

from evaluations.sql_spider.dataset import load_spider
from evaluations.sql_spider.grammar import (
    build_dynamic_sql_grammar,
    parse_db_info,
    extract_schema_identifiers,
)
from evaluations.sql_spider.metrics import SQLMetrics
from evaluations.sql_spider.generation import run_crane_csd, run_unconstrained
from evaluations.sql_spider.environment import setup_dafny_environment
from evaluations.sql_spider.executor import execute_accuracy, score_predictions

__all__ = [
    "load_spider",
    "build_dynamic_sql_grammar",
    "parse_db_info",
    "extract_schema_identifiers",
    "SQLMetrics",
    "run_crane_csd",
    "run_unconstrained",
    "setup_dafny_environment",
    "execute_accuracy",
    "score_predictions",
]
