# AGENTS.md — `synthesis/evaluate/grammars/`

## Scope

**Lark grammar** sources shared by constrained decoding and benchmarks (GSM, SQL, SMILES classes, etc.).

## Rules

- Grammars drive **incremental parsers** and DFA masks; edits affect validity, performance, and verifier-facing contracts—test downstream evaluation after changes.
- Prefer **additive** changes (new files, new start rules) over silent rewrites of shared productions unless coordinated with **`verify/library/`** and all consumers.

## See also

- **`README.md`** in this folder for the file index.
