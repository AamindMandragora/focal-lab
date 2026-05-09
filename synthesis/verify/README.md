# Verify Stage

The verify stage converts generated strategy text into a trustworthy executable artifact.
It has two tightly coupled jobs: formal verification and compilation.

## Responsibilities

- Verify full Dafny source against required contracts.
- Parse and surface verification diagnostics for refinement prompts.
- Compile verified Dafny code to Python modules.
- Return deterministic artifacts used by runtime evaluation.

## Key Files

- `verifier.py`
  - Dafny verification wrapper and diagnostics parsing.
- `compiler.py`
  - Dafny build wrapper (`--target:py`) with output capture and error parsing.
- `library/`
  - Dafny source files used as synthesis substrate; see `library/README.md` for a member-by-member index of `VerifiedAgentSynthesis.dfy`.

## Verification and Compile Relationship

Compilation is intentionally grouped here because:

- It only runs after verification succeeds.
- It depends on the same source assembly process and include paths.
- Verification/compile failures are part of one Dafny-centric refinement loop.

## Output Expectations

Compiled outputs are staged by the pipeline under each run directory in:

- `outputs/generated/<run_id>/python/`

The corresponding Dafny source snapshot is preserved in:

- `outputs/generated/<run_id>/dafny/`
