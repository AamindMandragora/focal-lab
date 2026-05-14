# AGENTS.md — `synthesis/verify/library/`

## Scope

**Dafny** template, primitive library, and contracts that define admissible synthesized strategies (`GeneratedCSD.dfy`, `VerifiedAgentSynthesis.dfy`, etc.).

## Rules

- **Contracts are authoritative:** pre/postconditions and lemmas are not optional documentation; changing them requires proof updates and explicit intent.
- Generated strategy bodies plug into a **fixed template**; keep extension points and `extern` axioms aligned with **`evaluate/`** runtime behavior.
- `AppendTaskGuidance` is a runtime prompt-policy hook with zero Dafny cost;
  keep its contract aligned with the shared LM runtime implementation.
- Coordinate with **`generate/`** if template placeholders or allowed call patterns change.

## See also

- **`README.md`** in this folder for file map, per-member summaries of `VerifiedAgentSynthesis.dfy`, and editing guidance.
- **`../reference/README.md`** for verified baseline-style example strategies (`crane.dfy`, `itergen.dfy`, `cars.dfy`).
