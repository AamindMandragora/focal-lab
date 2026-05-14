# AGENTS.md — `synthesis/evaluate/benchmarks/gsm_symbolic/`

## Scope

**GSM-Symbolic** evaluation: prompts, extraction, arithmetic/symbolic equivalence, and constrained-window syntax checks.

## Rules

- Keep scoring and delimited-span parsing rules aligned with **`eval_logic.py`** and any grammar assets under **`evaluate/grammars/`** referenced here.
- `generation.py` should reset shared LM task-guidance state before each
  example so `AppendTaskGuidance` applies from the start of that invocation.
- Do not encode synthesis **strategy advice** in prompts (root **`AGENTS.md`**).

## See also

- **`README.md`** in this folder.
