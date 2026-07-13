# Baseline helper-coverage audit — can metaDecode re-generate every baseline CSD?

**Date:** 2026-07-01
**Question (user):** Are metaDecode's current author-visible helpers sufficient to
re-generate all the baseline CSDs we compare against (CRANE / IterGen / CARS)? Any
missing helper would be an easy gap to clear toward a win.

## Method

- **Inputs:** current author menu `_ALL_HELPER_NAMES` (from `generate/prompts.py`,
  = scraped `TOOL_REFERENCE` minus `_MENU_PRUNED_HELPERS`), and the hand-written
  baseline reconstructions in `verify/reference/*.dfy`.
- **Output:** per baseline, every `helpers.X(` / `CSDHelpers.X(` primitive it calls,
  classified as in-menu / pruned-from-menu / absent-from-library.
- **Algorithm:** grep each reference `.dfy` for helper calls → check membership in
  the live menu set and the pruned set (done by importing prompts.py on focal).
- Read-only. $0 (no author-model calls, no eval jobs).

## Result — 100% primitive coverage, ZERO gaps

Author menu size = **65**. Every distinct helper used by every baseline
reconstruction is in the author menu:

| Baseline (file) | Helpers used | Coverage |
|---|---|---|
| CARS / SMILES (`cars.dfy`) | AppendConstrainedToken, CloseConstrainedSpan, ConstrainedStep, SafePenalizedConstrainedStep, SoftConstrainedStep, UnconstrainedStep | all in menu |
| CRANE / GSM (`crane.dfy`) | AppendConstrainedToken, CloseConstrainedSpan, GroupBoostedConstrainedStep, UnconstrainedStep | all in menu |
| CRANE one-call (`crane_faithful.dfy`) | CraneGeneration | in menu |
| IterGen / Spider (`itergen.dfy`) | AppendConstrainedToken, CloseConstrainedSpan, SafeSoftConstrainedStep, UnconstrainedStep | all in menu |
| unconstrained (`unconstrained.dfy`) | UnconstrainedStep | in menu |
| gcd (`gcd.dfy`) | AppendConstrainedToken, CloseConstrainedSpan, ConstrainedStep, OpenConstrainedSpan | in menu |

Nothing pruned-out, nothing absent. (`helpers.cost` is a field, not a helper —
excluded.)

## What this proves — and what it does NOT

**Proves (static, necessary condition):** there is NO "missing helper" gap. The
author model can literally *write* every primitive each baseline needs. So if
metaDecode loses a cell, it is NOT because a primitive is unavailable — adding a
helper is not the lever.

**Does NOT prove (two honest caveats):**
1. **Reconstruction ≠ scored baseline.** These `.dfy` files are metaDecode's
   hand-written re-expressions of the baselines. The baseline NUMBERS we report
   come from running the real baseline implementations (CRANE/IterGen repos,
   `run_legacy_fixed_strategy.py` for CARS) — not from these `.dfy`. Expressibility
   of the reconstruction is weaker than behavioral equivalence to the scored
   baseline.
2. **Coverage ≠ discovery.** Having the primitives on the menu does not mean the
   author model will *compose them correctly* in a cold run. That is a search /
   feedback-loop / reasoning question, not a helper-set question.

## Implication (where the real levers are)

Since primitive coverage is complete, a lost cell points at composition/discovery,
not a missing helper. The next-level check that would actually test caveats 1 & 2:

- **$0 empirical re-eval** of the three reconstructions (`crane_faithful.dfy`,
  `cars.dfy`, `itergen.dfy`) via `--initial-strategy-file ... --max-iterations 1`
  (pure re-eval, no author/Bedrock calls, local eval model on GPU only). Compare
  each reconstruction's score to the recorded baseline number:
  - **Matches** → primitives AND a valid composition both exist; the gap is purely
    the author *finding* it → invest in feedback signals / author reasoning.
  - **Doesn't match** → the reconstruction is not behaviorally faithful; the
    mismatch localizes exactly which step to fix (and whether a helper *behaves*
    differently than the baseline even though its name is present).

This re-eval is gated on a user go-ahead (it launches GPU eval jobs).
