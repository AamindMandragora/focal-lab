# Reference CSD strategies (Dafny)

Verified **formal specifications** that mirror published strategy families (GCD/SynCode, CRANE, IterGen, CARS) using only `CSDHelpers` from `library/VerifiedAgentSynthesis.dfy`.

> **These files are for verification and paper illustration only.** Baseline evaluation uses the legacy external codebases (`legacy/CRANE`, `legacy/itergen`, `legacy/cars`) and vendored SynCode (`synthesis/evaluate/syncode/`) via `run_legacy_fixed_strategy.py`. Do not compile these reference files for evaluation.

Each file defines its own module (`ReferenceGcdCSD`, `ReferenceCraneCSD`, etc.) so they can be verified together without duplicate `GeneratedCSD` names.

For a concise index of every `LM`, `Parser`, and `CSDHelpers` member, see [`../library/README.md`](../library/README.md).

## Files

| File | Role |
|------|------|
| `gcd.dfy` | **Greedy Constrained Decoding (SynCode-style).** Immediately opens a `<<` span and hard-masks every token; no unconstrained reasoning, no group boosting, rollback, or adaptivity. Closes with `>>` when the parse is complete. |
| `crane.dfy` | **CRANE-style.** Unconstrained prefix; inside `<<`…`>>`, `GroupBoostedConstrainedStep` with empty groups (hard mask only). |
| `itergen.dfy` | **IterGen-style.** Same outer loop; inside span, `ConstrainedSymbol` with stable-prefix rebuild (chunk + longest valid prefix). |
| `cars.dfy` | **CARS-style.** Same outer loop; inside span, `AdaptiveConstrainedStep` with `validTokenGroups` and `stepTokenBudget` as narrow threshold. |

## Design distinction: GCD vs CRANE

GCD forces constrained mode from the first token (`OpenConstrainedSpan` immediately, all model-chosen tokens are grammar-masked). CRANE allows unconstrained reasoning before `<<` and uses adaptive switching. On benchmarks like Spider/SMILES where the output is a formal expression, CRANE can reason ("I need to join these tables...") before emitting `<<SELECT ...>>`, while GCD constrains from token 1.

## Verify

From the repository root:

```bash
dafny verify synthesis/verify/reference/gcd.dfy \
               synthesis/verify/reference/crane.dfy \
               synthesis/verify/reference/itergen.dfy \
               synthesis/verify/reference/cars.dfy
```
