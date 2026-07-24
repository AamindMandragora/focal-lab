# AGENTS.md — `synthesis/evaluate/benchmarks/`

## Scope

**Pluggable benchmarks** (datasets, metrics, prompts, parsers) selected via registry.

## Rules

- Each benchmark lives in its own package with **`eval_logic.py`** as the delegation surface expected by **`evaluator.py`**.
- New benchmarks: add a subfolder, register in **`registry.py`**, and add **`README.md`** / **`AGENTS.md`** there.
- Keep evaluation semantics documented next to the code that implements them.

## See also

- **`README.md`** in this folder for layout and registry conventions.
