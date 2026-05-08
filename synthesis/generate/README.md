# Generate Stage

The generate stage is responsible for producing candidate Dafny strategy bodies that can be inserted into the strategy template and passed through formal verification.

## Responsibilities

- Define and maintain synthesis prompts.
- Generate initial strategy candidates from the user task.
- Refine strategies when verification, compilation, runtime, or evaluation fails.
- Preserve model-facing rationale blocks when required by project conventions.

## Key Files

- `generator.py`
  - Main generation/refinement orchestration.
- `prompts.py`
  - Prompt templates used for initial generation and iterative refinement.
- `rationale.py`
  - Utilities for extracting or normalizing rationale sections embedded in strategy text.

## Input and Output Contract

Input:

- Task description from CLI.
- Structured failure feedback from later stages.

Output:

- Strategy body text compatible with insertion into
  `synthesis/verify/library/GeneratedCSD.dfy`.

## Important Constraint

This repository treats synthesis as a controlled study.
Prompt content must avoid hidden strategy coaching and should remain limited to allowed task/tool context as described in `AGENTS.md`.
