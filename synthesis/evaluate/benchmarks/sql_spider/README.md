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

Fixed-strategy prompts come from `../prompt_profiles/sql.yaml`. GCD and IterGen
both select its byte-identical `direct` profile, which preserves IterGen's
original `quey` typo and trailing space. CRANE selects the same file's
`chain_of_thought` profile, retaining the same database fields and question
while requesting reasoning and a final `<< >>` query. Token-0 SQL decoding uses
hidden constrained chunks; the evaluator still accepts visible spans from
older outputs and raw SQL from direct outputs.

## Runtime Notes

- Spider evaluation is execution-grounded: generated SQL is executed and compared against gold-query behavior.
- The benchmark includes vendored evaluator dependencies under `syncode` support paths and benchmark utilities.
