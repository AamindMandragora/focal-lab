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

Strategies decide their own constraint behaviour. The prompt appends `<< >>` delimiter instructions — CRANE can reason before emitting `<<SMILES>>`, while GCD constrains from token 1. The evaluator prefers extracting from `<< >>` when present and falls back to `clean_smiles_output` on the raw text otherwise.

## Evaluation Behavior

SMILES scoring treats syntax validity and class-membership quality as separate signals.
The synthesis loop can use these diagnostics to discourage degenerate strategies that exploit formatting without producing meaningful molecules.

Multi-sample evaluation (synthesis eval and legacy baselines) updates each class prompt with empirical context from the current run:

- **Good results:** novel, syntax-valid, in-class molecules (unique scoring numerator).
- **Bad results:** prior failed or duplicate attempts listed so deterministic decoders do not repeat the same output.

State is managed in `prompt_state.py` and applied through `eval_logic.py` hooks used by `evaluator.py` and `run_legacy_fixed_strategy.py`.
