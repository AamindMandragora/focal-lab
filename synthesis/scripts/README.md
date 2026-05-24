# `synthesis/scripts/`

Optional **maintenance and ablation** scripts that drive `python -m synthesis.run_synthesis` (or inspect `outputs/generated/`) from the repository root.

They are not imported by the core package at runtime; run them explicitly with `PYTHONPATH` set to the repo root when documented in each script's docstring.

## Contents

- **`ablation_beam_bandit.py`** — Grid search over refinement beam size and helper-selection policy.
- **`reevaluate_compiled_csd.py`** — Re-run evaluation on an already-compiled GeneratedCSD.py.
- **`collect_paper_results.py`** — Collect baseline and synthesis results into paper-ready LaTeX table fragments. Reads `outputs/baselines/` and `outputs/generated/`, emits main results + ablation tables. Main/metaDecode rows use the **`opus4.7`** profile by default; **`--gen-profiles`** defaults to **`opus4.7,gpt5.4,gemini-pro`** for the synthesizer-model ablation table. Use **`--paper-main-table`** / **`--paper-bold-best`** to print Table~1 rows for `paper/experiments.tex`, or **`--write-paper`** to patch the marked block in that file automatically. Pass **`--git-tracked-only`** to include only metrics whose source `outputs/**/*.json` paths are tracked by git (cells without such JSON emit `\todo{--}`).
- **`split_smiles_class_baselines.py`** — One-time repair: split combined SMILES baseline JSONs (all classes in one file) into per-class files with recomputed accuracy/syntax metrics. Use **`--dry-run`** then **`--apply --backup`**.

Scripts are self-contained CLIs. See each file's module docstring for arguments and examples.

## See also

- **`AGENTS.md`** in this folder for agent constraints when adding or editing scripts.
