# `synthesis/scripts/`

Optional **maintenance** scripts run from the repository root with `python -m synthesis.scripts.<name>`.

They are not imported by the core package at runtime.

## Contents

- **`reevaluate_compiled_csd.py`** — Re-run evaluation on an already-compiled `GeneratedCSD.py` (used by the matrix after metadecode synthesis).
- **`collect_paper_results.py`** — Aggregate baseline and synthesis JSONs into LaTeX table fragments from `outputs/`.
- **`plot_step_budget_baselines.py`** — Plot accuracy and per-example runtime vs `max_steps` from cached baseline JSONs (`outputs/plots/step_budgets/` by default).
- **`report_legacy_upstream_diff.py`** — Compare patched `legacy/*` trees against upstream for patch maintenance.

See each module docstring for CLI flags.

## See also

- **`AGENTS.md`** in this folder.
- **`outputs/README.md`** — artifact layout (`model/benchmark/strategy`).
