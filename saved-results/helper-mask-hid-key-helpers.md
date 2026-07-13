# Helper mask was hiding the key helpers from the strategy author

**Date:** 2026-06-11 (overnight autonomous session)

## What this is

The root cause behind ~40+ failed GSM-1.5B synthesis attempts (and a likely contributor to
the acrylates-1.5B failures): the adaptive helper mask (`--adaptive-helper-mask
--helper-selection-policy bandit`) was pruning the helper documentation shown to the
strategy author down to **3 of 49 prunable helpers** — and the pruned set included exactly
the helpers that past diagnoses identified as the needed levers.

## The evidence chain

- **Inputs:** loopint2 run (`gsm1p5b_seed123_loopint2_20260611`), whose task text explicitly
  told the author to detect `<<` "at the STRING level - an unconstrained chunk that stops
  when the generated text reaches '<<'".
- **Observed output:** across all evaluated attempts, no strategy ever called
  `UnconstrainedChunk` (the helper with exactly that string-stop behavior — and the
  documented May fix for the span-entry tokenization bug). Authors used `UnconstrainedStep`
  + token-equality triggers instead, which fire ~1/49 examples because the tokenizer splits
  `<<` into two `<` tokens. Two attempts even *mentioned* UnconstrainedChunk in their
  rationale comments, then didn't call it.
- **The step that failed:** `feedback_loop.py` puts `UnconstrainedChunk` in
  `PRUNABLE_HELPERS` (line ~488). With the bandit mask active and "kept 3/49 prunable
  helpers", the helper was absent from the author-visible surface, so the author couldn't
  (or wouldn't) call it. Same for `DeadEndAvoidingStep` / `RollbackAndRegenerate` in the
  SMILES acrylates runs (0 mentions in any of 30+ attempt strategies across 3 runs).
- Past WINNING strategies (gsm7b_seed429, spider1p5b_nondisjoint) call `UnconstrainedChunk`
  directly — the helper is essential, verified, and battle-tested.

## The fix

Launch with `--no-adaptive-helper-mask` so the full helper surface is documented for the
author. Applied in:
- `.focal_edit/gsm1p5b_loopint3.sh` (GSM-1.5B, GPU 1, warm from loopint attempt 9 at
  40.8%/57.1%, dual-mechanism guidance kept)
- `.focal_edit/fixAB_acr1p5B_retry4.sh` (acrylates-1.5B, GPU 2, warm from retry-1 attempt
  14, diversity-mechanics guidance kept)

Both launched 2026-06-11 ~05:00 local (22:59 UTC 06-10).

## How to check this in any future stuck run

1. `grep "kept .*prunable" <log>` — if the mask is keeping a small fraction, be suspicious.
2. Compare helper names in attempt rationales vs actual `helpers.X` calls in strategy code —
   an author that PLANS a helper but never CALLS it is a strong signal the helper isn't on
   its documented surface.
3. `PRUNABLE_HELPERS` vs `NON_PRUNABLE_HELPERS` sets live in
   `synthesis/evaluate/feedback_loop.py` (~line 445).

## Open question for daytime

Whether `--adaptive-helper-mask` should remain the default in launchers at all. It exists
to focus the author, but it can silently starve the author of the right tool. The bandit
only learns a helper is good if some attempt uses it — but no attempt can use it while
it's masked (a cold-start trap with no escape).
