# AGENTS.md — `synthesis/evaluate/syncode/`

## Scope

**Vendored Syncode** tree (upstream grammar-constrained decoding / DFA mask store). Subpackages under `syncode/` are third-party code with local patches.

## Rules

- **Default:** do not edit vendored sources for product features; fix issues in first-party **`synthesis/evaluate/`** when possible.
- **When you must patch Syncode:** keep patches **minimal**, document rationale in the commit message, and prefer aligning with upstream over long forks.
- Every `SyncodeLogitsProcessor` must construct its `ConstraintAudit` before either unconstrained-fallback path can record into it; prompt resets preserve the run-level counter.
- `AdaptiveSynCode(start_inside_constrained=True)` is the minimal local support
  for delimiter-free benchmarks such as SMILES. It must reset the existing DFA
  grammar decoder at the prompt boundary and constrain the first generated
  token; delimiter-based behavior remains the default.
- Do not add **`README.md` / `AGENTS.md`** inside every nested database or package folder; this file plus existing Syncode READMEs are the contract.

## See also

- **`README.md`** in this folder (vendoring notes).
- Upstream layout: `syncode/syncode/`.
