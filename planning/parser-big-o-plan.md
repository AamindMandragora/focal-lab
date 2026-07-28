# Parser helper big-O plan

**Task:** Make CSD parser helper calls cheaper, starting with SMILES completeness checks, then audit sibling helper paths for repeated whole-prefix work.

## Inputs

- Runtime parser bridge: `synthesis/evaluate/benchmarks/common/parser_utils.py`
- Runtime masking helper: `synthesis/evaluate/benchmarks/common/model_utils.py`
- Current issue: `IsValidPrefix` and `IsCompletePrefix` are called many times during constrained decoding. They currently rebuild the prefix text before cache lookup; completeness also uses full Lark parse on cache miss.

## Outputs

- Tests that fail on the current code and pass after the change.
- A parser helper change that keeps semantics the same but lowers repeated whole-prefix work.
- A short audit of other helper methods with clear next optimization targets.

## Algorithm

1. Add a focused test using fake Dafny-token sequences and fake parser components.
2. Verify the test fails against the current helper.
3. Add prefix-level caching before text conversion for `IsValidPrefix`, `IsCompletePrefix`, `ValidNextTokens`, `ValidNextTokenCount`, and `ValidNextToken`.
4. Add an incremental/accept-state completeness fast path where safe; keep full parse as correctness fallback.
5. Re-run the focused test and a small parser/helper subset.
6. Audit helper timing paths and record what remains O(prefix length).
7. For the next optimization pass, profile before editing and keep the stage order:
   - SMILES stage: after H86 is recorded, check `GenerateLogits.prefix_text` in `model_utils.py` on the next SMILES run.
   - GSM/Spider stages: check `CompletedSchemaSymbolCount` only when schema-symbol rollback helpers are active again.
   - Any stage: check `GetTopKTokens` only if the chosen strategy uses top-k helpers heavily.

## Safety notes

- The first parser patch was copied into focal main after explicit user instruction; already-running H86 may still keep old imported code.
- Future optimization patches should use an isolated focal worktree first, then copy into focal main only after focused red/green tests and nearby smoke tests pass.
- Treat this as a speed-only change; no SMILES/GSM/Spider score should change if semantics are preserved.
