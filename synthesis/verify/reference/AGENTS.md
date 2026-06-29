# AGENTS.md — `synthesis/verify/reference/`

## Scope

**Verified Dafny example strategies** (CRANE, GCD, IterGen, CARS, unconstrained) for formal verification and **`run_reference_strategy`** baselines.

## Rules

- These are **contract examples**, not synthesis prompts; do not embed benchmark answer hints.
- Keep bodies aligned with **`../library/VerifiedAgentSynthesis.dfy`** helper contracts.
- Legacy fixed baselines use **`legacy/`** Python repos, not these files — do not conflate the two paths in docs or tooling.
- Each file uses a distinct `Reference*CSD` module name for joint `dafny verify`; **`run_reference_strategy`** rewrites to `GeneratedCSD` before compile.

## See also

- **`README.md`** in this folder.
- **`../library/AGENTS.md`**.
