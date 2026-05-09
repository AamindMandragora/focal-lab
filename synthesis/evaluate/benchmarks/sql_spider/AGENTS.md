# AGENTS.md — `synthesis/evaluate/benchmarks/sql_spider/`

## Scope

**Spider** text-to-SQL evaluation: schema formatting, gold execution comparison, and SQL constrained decoding.

## Rules

- Respect path overrides (`SPIDER_*` env vars) documented in root **`README.md`**; keep DB and table metadata loading robust.
- Execution accuracy and SQL syntax validity belong here or in delegated helpers, not in **`evaluator.py`** as one-off branches.

## See also

- **`README.md`** in this folder.
