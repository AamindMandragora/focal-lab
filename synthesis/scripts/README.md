# `synthesis/scripts/`

Optional **maintenance and ablation** scripts that drive `python -m synthesis.run_synthesis` (or inspect `outputs/generated/`) from the repository root.

They are not imported by the core package at runtime; run them explicitly with `PYTHONPATH` set to the repo root when documented in each script’s docstring.

## Contents

- Scripts are self-contained CLIs (e.g., grid searches over beam size and helper-selection policy). See each file’s module docstring for arguments and examples.

## See also

- **`AGENTS.md`** in this folder for agent constraints when adding or editing scripts.
