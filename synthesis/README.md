# Synthesis Package

The `synthesis/` package is the operational center of the repository.
It provides an end-to-end loop that produces candidate CSD strategies, proves correctness properties in Dafny, compiles those strategies to executable Python modules, evaluates them on target benchmarks, and feeds failures back into the next generation attempt.

## Top-Level Modules

- `run_synthesis.py`
  - Main CLI entry point for iterative synthesis.
  - Configures models, thresholds, evaluation settings, and output layout.
- `project_defaults.py`
  - Centralized defaults for local paths (Dafny binary, CRANE/Spider resources, etc.).

## Stage Subpackages

- `generate/`
  - Prompt construction and strategy body generation/refinement.
- `verify/`
  - Dafny verification and Dafny-to-Python compilation wrappers.
- `evaluate/`
  - Runtime environment setup, benchmark evaluation, parser integration, and feedback-loop orchestration.

## Canonical Stage Flow

The package is intentionally structured to mirror the pipeline order:

1. `generate`
2. `verify`
3. `evaluate`

Compile is implemented under `verify` because compilation is only valid after verification and shares Dafny-specific lifecycle code.

## Artifact Ownership

- Successful and failed run artifacts are written under repo-root `generated/`.
- Baseline experiment artifacts are written under repo-root `baselines/`.
- `synthesis/` itself contains implementation, not experiment outputs.

## Design Philosophy

- Keep benchmark-specific behavior in benchmark packages, not in generic pipeline code.
- Keep parser/runtime performance guarantees explicit (Syncode DFA masks, caching, parser factories).
- Keep each synthesis attempt auditable through saved reports and run-local artifacts.
