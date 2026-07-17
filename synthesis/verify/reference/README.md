# Reference CSD strategies (Dafny)

Verified **formal specifications** that mirror published strategy families (Unconstrained, GCD/SynCode, CRANE, IterGen, CARS) using only `CSDHelpers` from `library/VerifiedAgentSynthesis.dfy`.

> **These files are for verification and paper illustration only.** Baseline evaluation uses the legacy external codebases (`legacy/CRANE`, `legacy/itergen`, `legacy/cars`) and vendored SynCode (`synthesis/evaluate/syncode/`) via `run_legacy_fixed_strategy.py`. Do not compile these reference files for evaluation.

Each file defines its own module (`ReferenceGcdCSD`, `ReferenceCraneCSD`, etc.) so they can be verified together without duplicate `GeneratedCSD` names.

For a concise index of every `LM`, `Parser`, and `CSDHelpers` member, see [`../library/README.md`](../library/README.md).

## Files

| File | Role |
|------|------|
| `unconstrained.dfy` | **Unconstrained generation.** Pure `UnconstrainedStep` loop with no grammar enforcement. Serves as the lower-bound baseline for syntax validity and upper-bound reference for unconstrained model capability. |
| `gcd.dfy` | **Greedy Constrained Decoding (SynCode-style).** Immediately opens a `<<` span and hard-masks every token; no unconstrained reasoning, no group boosting, rollback, or adaptivity. Closes with `>>` when the parse is complete. |
| `crane.dfy` | **CRANE-style.** Unconstrained prefix; inside `<<`…`>>`, `GroupBoostedConstrainedStep` with empty groups (hard mask only). |
| `itergen.dfy` | **IterGen-style.** Same outer loop; inside span, per-token `SafeSoftConstrainedStep` with zero boost (sample unconstrained, check grammar, fall back to hard mask if invalid — matching `_get_next_token_grammar`). |
| `cars.dfy` | **CARS-style (full adaptive rejection sampling).** `ConstrainedStep` for the first constrained token (`constrain_first`); `SoftConstrainedStep` with zero boost for exploration (unconstrained, like new trie nodes); on grammar violation the attempt is rejected, the failing token is penalised, and the span is rolled back; retries use `SafePenalizedConstrainedStep` (hard mask + accumulated penalties, like revisited trie nodes with `log_theta`). |

## Design distinction: GCD vs CRANE

GCD forces constrained mode from the first token (`OpenConstrainedSpan` immediately, all model-chosen tokens are grammar-masked). CRANE allows unconstrained reasoning before the constrained expression and uses adaptive switching. On benchmarks where the output is a formal expression, CRANE can reason before entering the constrained expression, while GCD constrains from token 1. Current Spider and SMILES evaluation prompts ask for bare SQL or bare SMILES rather than visible `<< >>` spans.

## Verify

From the repository root:

```bash
dafny verify synthesis/verify/reference/unconstrained.dfy \
               synthesis/verify/reference/gcd.dfy \
               synthesis/verify/reference/crane.dfy \
               synthesis/verify/reference/itergen.dfy \
               synthesis/verify/reference/cars.dfy
```
