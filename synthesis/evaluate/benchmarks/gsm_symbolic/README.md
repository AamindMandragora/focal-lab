# GSM-Symbolic Benchmark

This module evaluates synthesized CSD strategies on GSM-Symbolic style arithmetic reasoning tasks with constrained span behavior.

## Responsibilities

- Load GSM examples (including local CRANE-style JSON sources when configured).
- Build prompt text aligned with GSM reasoning expectations.
- Configure grammar-aware runtime environment for constrained decoding.
- Score model outputs with symbolic/numeric extraction and equivalence logic.

## Key Files

- `dataset.py`: dataset loading, metadata enrichment, split utilities.
- `prompts.py`: prompt formatting helpers.
- `grammar.py`: grammar adaptation helpers for dynamic variable restrictions.
- `generation.py`: benchmark generation wrappers used by evaluator.
  Resets task-guidance state before each example and records accepted guidance
  after the compiled CSD runs.
- `environment.py`: runtime setup for compiled Dafny strategy execution.
- `metrics.py`: GSM-oriented scoring and metrics utilities.

## Notes

- Variable-aware grammar specialization is used for faithful constrained decoding.
- This benchmark is the primary synthesis target in current workflows.
- Local CRANE JSON loads use the symbolic template (`question_parsed`) as the primary `question` field; instantiated prose is kept as `question_instantiated`. HuggingFace rows prefer `question_parsed`, then `original_question`, then `question` when building prompts.
