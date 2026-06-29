# SynCode evaluation harnesses (vendored)

Upstream benchmark evaluation modules (`code_eval`, `math_eval`, `sql_eval`, etc.) bundled with the vendor drop.

## What Metadecode uses

**`run_legacy_fixed_strategy.py`** imports **`infer.py`**, which loads these modules at import time for GCD/CRANE baseline paths. Synthesis evaluation does **not** call them directly.

Prefer fixing behavior in first-party **`synthesis/evaluate/`** code when possible; patch here only when the baseline SynCode entry path requires it.

## See also

- **`../../README.md`** — vendoring overview.
- **`../../AGENTS.md`** — edit policy.
