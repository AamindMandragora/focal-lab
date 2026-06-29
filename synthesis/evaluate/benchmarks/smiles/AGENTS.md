# AGENTS.md — `synthesis/evaluate/benchmarks/smiles/`

## Scope

**SMILES** class-constrained molecular string benchmark: prompts, class-specific grammars, RDKit-backed checks.

## Rules

- Evaluation uses **one class per run** for grammar binding (see CLI `--smiles-classes`); keep **`eval_logic.py`** and **`dataset.py`** consistent with that contract.
- Grammar text and assets under **`data/`** must stay in sync; update **`data/AGENTS.md`** when adding classes or files.

## See also

- **`README.md`** and **`data/README.md`**.
