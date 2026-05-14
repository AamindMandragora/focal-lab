# AGENTS.md — `synthesis/evaluate/benchmarks/common/`

## Scope

**Shared benchmark helpers** (e.g., parser construction, formatting) used by multiple tasks.

## Rules

- Prefer **small, reusable** utilities; avoid importing benchmark-specific modules from sibling folders in ways that create cycles.
- Parser helpers must stay compatible with **DFA-mask** incremental parsing used elsewhere in evaluation.
- Runtime LM prompt-guidance state should stay benchmark-agnostic and
  first-call-wins so evaluation metrics remain interpretable.

## See also

- **`README.md`** in this folder.
