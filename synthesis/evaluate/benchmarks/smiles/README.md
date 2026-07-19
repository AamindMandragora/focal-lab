# SMILES Benchmark

This module evaluates synthesized CSD strategies on constrained molecular-string generation tasks.

## Responsibilities

- Load class-specific SMILES tasks and prompts.
- Apply class-dependent grammar constraints.
- Evaluate generated candidates for syntax validity and class membership.
- Report paper-aligned metrics plus anti-degeneracy diagnostics.

## Key Files

- `dataset.py`: class/task loading and prompt assembly.
- `generation.py`: generation wrappers for evaluator integration.
- `environment.py`: runtime setup for compiled strategy execution.
- `metrics.py`: RDKit-based validity and membership metrics.
- `data/`: per-class assets (grammars, exemplars, reference sets).

## Constraint mode

Fixed-strategy prompts come from the single shared
`../prompt_profiles/smiles.yaml` file. GCD and IterGen both select its `direct`
profile; CRANE selects its `chain_of_thought` profile. Both finish at
`Molecule:` and request a bare SMILES answer with no visible `<< >>`
delimiters. Grammar-guided SMILES generation uses hidden constrained chunks;
the evaluator still prefers an older visible span when present and otherwise
cleans the raw output.

## Evaluation Behavior

SMILES scoring treats syntax validity and class-membership quality as separate signals.
The synthesis loop can use these diagnostics to discourage degenerate strategies that exploit formatting without producing meaningful molecules.
