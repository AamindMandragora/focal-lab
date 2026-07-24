# Benchmarks Package

This package contains dataset-specific evaluation implementations used by synthesis.
Each benchmark module owns its own dataset loading, prompt formatting, answer extraction/scoring assumptions, and runtime environment details when needed.

## Subpackages

- `common/`
  - Shared utilities for model loading/runtime wrappers and parser integration.
- `gsm_symbolic/`
  - GSM-Symbolic reasoning benchmark implementation.
- `sql_spider/`
  - Text-to-SQL Spider benchmark implementation.
- `smiles/`
  - Molecular-string generation benchmark implementation.

## Architecture Rule

Generic orchestration lives in `synthesis/evaluate/evaluator.py` and `feedback_loop.py`.
Benchmark-specific logic should stay in benchmark packages so the pipeline remains modular and testable.
