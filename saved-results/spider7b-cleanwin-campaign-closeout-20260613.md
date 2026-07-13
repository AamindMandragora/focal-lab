# Spider-7B clean-win campaign — closeout (both cold runs failed)

**Date:** 2026-06-13
**What:** The PhD-advisor-requested push to turn the Spider-7B cell from a 1-example *tie*
(ours 65.3% / IterGen 65.7% on held-out 300) into a *clean win*. Two cold synthesis runs were
launched with framework upgrades (IterGen-parity unit-rewind helper, 300s/example timeout,
timeout-as-failure fix). Both ended NO_ACCEPT. This documents why the campaign is exhausted.

## The two runs

| Run | Acceptance eval | Best result | Verdict |
|---|---|---|---|
| #1 (`spider7b_cleanwin_cold_20260612`) | N=100 (fixed first-100 of sorted train indices) | accepted att8 at 68.0/98.0 → **held-out 167/300 = 55.7%** | overfit to a fixed, non-representative third of the train pool |
| #2 (`spider7b_cleanwin_cold2_20260612`) | **N=300 (full train side)** | NO_ACCEPT, best valid att17 = **62.3% (187/300) / 96.7%** | honest train ceiling ~62% |

## Why this closes the campaign

- Run #1's accepted 68% was an artifact of evaluating on the same fixed first-100 train
  indices every attempt (the determinism audit proved this subset is deterministic, not a
  random draw — its difficulty mix is lighter than the full side). Held-out it fell to 55.7%.
- Run #2 removed that artifact by scoring every attempt on the full train-300. The strategy
  family's honest ceiling is **~62% accuracy** (8 valid attempts cluster 57–62% at 95–97%
  syntax; the rest are 0%/0% "fast-but-unclosed" — EOS before `<<` — or Dafny verification
  failures). 62% train will not beat IterGen's 65.7% held-out.
- This matches the earlier diagnosis (results_matrix rows 121 & 116): **96% of Spider-7B
  residual failures are in-schema semantically-wrong SQL** (valid queries that run and return
  the wrong rows — wrong join/aggregation/filter). Constrained decoding has leverage on syntax
  and schema-membership, **zero leverage on semantics.** The schema-rollback ceiling was
  computed at 59.0% < bar. IterGen's edge is schema-aware inference-time backtracking that
  shapes the whole decode trajectory toward correct joins — our strategy-synthesis framework
  has no fair equivalent.

## The see-saw also reproduced at 7B

Same failure mode as the 1.5B runs, milder: attempts either close spans slowly (57–62% acc,
high syntax) or emit EOS before `<<` and score 0/0. The closing pole's ceiling is ~62%.

## Helper-doc adoption test (secondary purpose of run #2)

Run #2 carried the rewritten `RegenerateUnitOnCheckFailure` documentation (the unit-rewind
helper). Whether the author adopted it this time is being checked by the run-#2/run-#3
post-mortems; if it was declined again, that is itself evidence the semantic-rewrite mechanism
doesn't help the 7B's confident-wrong-plan failure mode.

## Recommendation (for the user to rule on)

Accept the 65.3% tie as the final Spider-7B cell (user already accepted it once, 06-11). Two
honest cold runs now confirm fair CSD cannot clean-win this cell. The remaining theoretical
lever (k=10 exec-vote self-consistency, +26/300 projected) was ruled **unfair** earlier
(execution signal at inference IterGen lacks) and declined. No fair lever remains.

## Provenance

- Run #1: `outputs/generated/spider7b_cleanwin_cold_20260612/`, held-out rescore
  `rescore_our_spider7b_heldout_20260612.py` (167/300). See
  `saved-results/spider7b-cleanwin-run1-heldout-fail.md`.
- Run #2: `outputs/generated/spider7b_cleanwin_cold2_20260612/`, log
  `spider7b_cleanwin2_20260612.log`. Bars 0.68/0.93, `--eval-sample-size 300`, 300s/example.
- Standing cell entry: results_matrix.md Spider-7B "FINAL: tie ACCEPTED by user 06-11" (65.3%).
