# SQL Spider Benchmark

This module evaluates synthesized CSD strategies on text-to-SQL tasks using the Spider benchmark.

## Responsibilities

- Load Spider examples and schema context.
- Build schema-aware prompting context.
- Run constrained decoding for SQL generation.
- Score predictions with execution-based matching.

## Key Files

- `dataset.py`: dataset loading and schema/context utilities.
- `grammar.py`: SQL grammar helpers.
- `generation.py`: generation wrappers integrated with evaluator.
- `environment.py`: runtime setup for compiled strategy execution.
- `executor.py`: execution-accuracy scoring against SQLite databases.
- `metrics.py`: aggregate metrics and reporting helpers.

## Constraint mode

The evaluator exposes two prompt surfaces. `format_prompt_expression_only` renders the flattened string prompt used by hard-mask / constrained decoders, while `format_prompt_chain_of_thought` returns the chat-style legacy CRANE prompt. Both paths instruct the model to wrap its SQL query in `<< >>` delimiters — CRANE can reason before emitting `<<SELECT ...>>`, while GCD constrains from token 1. The evaluator extracts the answer from `<< >>` when present and falls back to the raw first paragraph for unconstrained baselines.
Fixed IterGen pre-renders the flattened prompt through the Qwen3.5 chat
template with thinking disabled for that model only. Scoring and evidence keep
the original flattened prompt.

## Runtime Notes

- Spider evaluation is execution-grounded: generated SQL is executed and compared against gold-query behavior.
- The benchmark includes vendored evaluator dependencies under `syncode` support paths and benchmark utilities.
