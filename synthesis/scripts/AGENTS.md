# AGENTS.md — `synthesis/scripts/`

## Scope

Standalone **CLI utilities** under **`synthesis/scripts/`** (matrix helpers, re-evaluation).

## Rules

- Follow root **`AGENTS.md`**: scripts must not embed synthesis **strategy guidance** in prompts; they only forward CLI task text and flags to pipeline entry points.
- Prefer **thin wrappers** around **`run_synthesis`**, **`Evaluator`**, or **`run_legacy_fixed_strategy`**; duplicate as little benchmark logic as possible.
- Document usage in the script **module docstring** and keep **`README.md`** here updated when adding scripts.
- **`reevaluate_compiled_csd.py`** is required by **`run_all_tests.py`**; do not remove without updating the matrix launcher.

## See also

- **`README.md`** in this folder.
