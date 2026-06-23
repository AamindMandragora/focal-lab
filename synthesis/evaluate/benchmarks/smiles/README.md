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
- `metrics.py`: RDKit-based validity and membership metrics; **`grammar_valid_with_fallback`** for tier-then-base grammar checks.
- `data/`: per-class assets (grammars, exemplars, reference sets).

## Constraint mode

Strategies decide their own constraint behaviour. The prompt appends `<< >>` delimiter instructions — CRANE can reason before emitting `<<SMILES>>`, while GCD constrains from token 1. The evaluator prefers extracting from `<< >>` when present and falls back to `clean_smiles_output` on the raw text otherwise.

## Evaluation Behavior

SMILES scoring treats syntax validity and class-membership quality as separate signals.
The synthesis loop can use these diagnostics to discourage degenerate strategies that exploit formatting without producing meaningful molecules.

**Grammar validity:** extracted SMILES bodies are checked against the active prompt tier's grammar first. When that fails (common for tier-2 delimited grammars that expect a trailing `>>`), scoring retries with a closed tier variant and then with the class **base** body grammar (`base_grammar_text` from `dataset.py`). **`syntax_valid`** is true when grammar and RDKit (if installed) both pass.

Pooled evaluation samples up to **200** attempts per class and stops early once a class reaches **`DEFAULT_SMILES_POOLED_SUCCESS_TARGET`** (CLI: `--smiles-unique-syntax-valid-target`, legacy alias `--cars-success-target`) first-occurrence **unique RDKit-valid** molecules.

Scoring uses first-occurrence unique molecules only (excluding in-prompt exemplars and repeats):

- **Syntax rate** = unique syntax-valid / `success_target`
- **Accuracy** = unique syntax-valid and in-class / `success_target`

Prompt feedback:

- **Greedy strategies** (`unconstrained`, `gcd`, `crane`, `itergen`, metadecode synthesis eval): dynamic **Good results** / **Bad results** suffixes via `prompt_state.py`. Only extracted SMILES (from `clean_smiles_output` / delimited span parsing) are recorded; empty extraction adds `(invalid)` to bad results instead of raw completion text. Native headers require SMILES notation only (no IUPAC/systematic names). Bad suffixes use **Below are past mistakes — do not repeat them.** with one mistake per line (no inline `SMILES:` label). Repeating an extracted bad SMILES appends the capped prior completion under a **Response:** block plus `[repeat N]` so greedy runs cannot stall on an unchanged prompt.
- **Stochastic strategies** (`rs`, `cars`): constant prompt so sampling diversity (and the CARS oracle trie) are not perturbed between attempts.

State is managed in `prompt_state.py` and applied through `eval_logic.py` hooks used by `evaluator.py` and `run_legacy_fixed_strategy.py`. Aggregation lives in `pooled_eval.py` (`aggregate_smiles_pooled_scores`).
