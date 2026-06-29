# Vendored Spider SQL evaluator

Subset of the Spider official evaluation scripts used for **execution accuracy** in `benchmarks/sql_spider/executor.py`.

Loaded dynamically (not via the `syncode` package namespace) so only `evaluation.py` and its dependency `process_sql.py` are required at runtime.

Nested `evaluation_examples/` and other upstream docs are reference material only.

## See also

- **`../../AGENTS.md`** — vendored Syncode policy.
- **`benchmarks/sql_spider/executor.py`** — loader entry point.
