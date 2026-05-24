# Evaluate Stage

The evaluate stage executes compiled strategies on benchmark tasks and returns structured metrics used by the synthesis feedback loop.

## Responsibilities

- Build runtime environment around compiled Dafny output.
- Load benchmark datasets and grammar resources.
- Run constrained decoding evaluation for each sample.
- Compute benchmark metrics and synthesis gate checks.
- Emit rich diagnostics used for strategy refinement.

## Main Components

- `evaluator.py`
  - Core sample evaluation loop and orchestration.
  - Delegates benchmark-specific prompt/answer/parser/scoring logic to `benchmarks/*/eval_logic.py`.
- `feedback_loop.py`
  - Generate/verify/compile/evaluate orchestration with iterative refinement.
- `runner.py`
  - Runtime helper paths used by local smoke/runtime routines.
- `parser_utils.py`
  - Compatibility wrapper re-exporting canonical parser utilities.
- `benchmarks/`
  - Dataset-specific modules (GSM-Symbolic, SQL Spider, SMILES).
  - `benchmarks/registry.py` selects the benchmark logic module.
  - `benchmarks/*/eval_logic.py` contains benchmark-specific evaluation behavior for easier unit testing.
- `grammars/`
  - Lark grammar definitions used by constrained decoding.
- `syncode/`
  - Vendored Syncode dependency for DFA mask store + parser internals.

## Runtime Constraints

- The parser path depends on Syncode DFA-mask caching for practical performance.
- Evaluation backends currently support runtime modes that provide token-level control (`huggingface`, `vllm`).
- Runtime LM wrappers support `SetNonDeterministic` (greedy vs temperature-1
  sampling; resets each example) and `AppendTaskGuidance`: the first non-empty CSD
  guidance block is appended to the evaluator prompt for that example, later
  calls are ignored, and accepted guidance is surfaced in evaluation feedback.
- Output artifacts from this stage are saved under per-run `results/` folders in `outputs/generated/`.
- Baseline snapshots are JSON files in `outputs/baselines/` with:
  - `accuracy`, `syntax_rate`
  - `metrics` (counts, optional sums/means for `generation_seconds` / `num_tokens`, optional `run_wall_time_seconds` or evaluator totals)
  - optional top-level `metadata` for legacy runs (prompt tier, adapter id, decode caps)
  - `answers[]` per row:
    - `question` — normalized benchmark question (plain text, aligned across strategies)
    - `prompt` — full prompt sent to the model
    - `generated` — raw model completion suffix used for scoring
    - `extracted` — parsed answer used for accuracy/syntax checks
    - `correct`, `syntax_valid` — per-example booleans
    - `generated_answer` — legacy alias of `extracted` for older readers
    - optional `generation_seconds`, `num_tokens`
- Fixed-strategy GSM baselines use the local CRANE GSM source rows so
  `unconstrained`, `gcd`, `crane`, `itergen`, `cars`, and `rejection_sampling`
  are compared on the same questions.
- The GCD adapter uses Syncode DFA-mask decoding but keeps GSM-Symbolic generation scoped to expression bodies: it starts after `<<`, wraps the generated body for scoring, caps expression length, finalizes the longest parseable expression prefix, and restricts GSM variables to numeric placeholders observed in the evaluation sample.
- GCD and IterGen legacy adapters load **one** Hugging Face model per subprocess (`syncode_run_session.SyncodeRunSession` / IterGen grammar rebind). Per-example tier-1 grammars swap cached DFA mask decoders only; each decode resets parser state via SynCode/IterGen `start`/`reset` so prior prompts cannot leak into the next example. Subprocess exit clears GPU state between matrix jobs.
- GSM syntax checks use a numeric-only grammar when examples do not expose numeric symbolic variables; arbitrary identifiers such as `reasoning` must not pass syntax on instantiated GSM rows.
- The legacy CARS adapter runs through the same benchmark registry as the other fixed strategies and raises on failed runs so incomplete artifacts are not mistaken for valid zero-score baselines; it uses the same GSM grammar tightening as the GCD adapter (allowed variables or numeric-only, inferred from the evaluation batch) and expression-only prompts for gsm_symbolic, Spider, and SMILES like IterGen/GCD.
- SMILES fixed-strategy rows (CRANE subprocess and in-repo adapters) share **`SmilesPromptState`** for multi-sample prompts and **`grammar_valid_with_fallback`** for **`syntax_valid`** (tier grammar, then class base body grammar, then RDKit when installed). Re-baseline cached CRANE SMILES JSONs after pulling **`030-vas-smiles-prompt-state-grammar`**.
- Baseline exports may contain empty generated strings; those still count as answer rows. Corrupt fixed-strategy artifacts are the ones with no answer rows or missing `generated_answer` fields.
- Legacy rows that do not report a syntax boolean are treated as syntax-invalid unless the adapter can annotate them with benchmark parser checks before export.
- CRANE-backed GSM rows do not carry `variable_types`; the exporter infers numeric symbolic identifiers from `gold_answer` before applying the GSM syntax parser.
