# AGENTS.md — `synthesis/evaluate/benchmarks/`

## Scope

**Pluggable benchmarks** (datasets, metrics, prompts, parsers) selected via registry.

## Rules

- Each benchmark lives in its own package with **`eval_logic.py`** as the delegation surface expected by **`evaluator.py`**.
- New benchmarks: add a subfolder, register in **`registry.py`**, and add **`README.md`** / **`AGENTS.md`** there.
- Keep evaluation semantics documented next to the code that implements them.
- Fixed SQL and SMILES CSD prompts must come from `prompt_profiles/*.yaml`.
  Keep GCD and IterGen on the same `direct` profile and every CRANE mapping on
  `chain_of_thought`.

## See also

- **`README.md`** in this folder for layout and registry conventions.
