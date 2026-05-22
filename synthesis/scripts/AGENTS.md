# AGENTS.md — `synthesis/scripts/`

## Scope

Standalone **CLI utilities** under **`synthesis/scripts/`** (ablations, batch runners, report helpers).

## Rules

- **`collect_paper_results.py`**: main/metaDecode tables filter **`opus4.7`** (`DEFAULT_MAIN_GEN_PROFILE`); synthesizer ablation table uses **`--gen-profiles`** (default **`opus4.7,gpt5.4,gemini-pro`**). Use **`--write-paper`** to refresh the marked block in **`paper/experiments.tex`**.
- Follow root **`AGENTS.md`**: scripts must not embed synthesis **strategy guidance** in prompts; they only forward CLI task text and flags to **`run_synthesis`**.
- Prefer **thin wrappers** around existing pipeline entry points; duplicate as little benchmark logic as possible.
- Document usage in the script **module docstring** and keep **`README.md`** here updated when adding new scripts.

## See also

- **`README.md`** in this folder.
