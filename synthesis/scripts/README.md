# `synthesis/scripts/`

Optional **maintenance and ablation** scripts that drive `python -m synthesis.run_synthesis` (or inspect `outputs/generated/`) from the repository root.

They are not imported by the core package at runtime; run them explicitly with `PYTHONPATH` set to the repo root when documented in each script's docstring.

## Contents

- **`ablation_beam_bandit.py`** — Grid search over refinement beam size and helper-selection policy.
- **`reevaluate_compiled_csd.py`** — Re-run evaluation on an already-compiled GeneratedCSD.py.
- **`collect_paper_results.py`** — Collect baseline and synthesis results into paper-ready LaTeX table fragments. Reads `outputs/baselines/` and `outputs/generated/`, emits main results + ablation tables.

Scripts are self-contained CLIs. See each file's module docstring for arguments and examples.

## See also

- **`AGENTS.md`** in this folder for agent constraints when adding or editing scripts.
