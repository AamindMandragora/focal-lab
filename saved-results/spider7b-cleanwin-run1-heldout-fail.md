# Spider-7B clean-win run #1 — accepted on train, FAILED held-out (overfit)

**Date:** 2026-06-12
**What:** first cold synthesis run of the Spider-7B clean-win campaign (IterGen-parity
unit-rewind helper available, 300s per-example limit, timeout-as-failure fix active).

## Result

| Stage | Score | N | Notes |
|---|---|---|---|
| Train (accept) | 68.0 acc / 98.0 syn | 100 (draw from train-300) | accepted at attempt 8/20, exactly at bar 0.68/0.93 |
| **Held-out (official grader)** | **55.7% (167/300)** / 98.3 syn | 300 (test side, all evaluated) | needed ≥198/300; pipeline scorer agrees exactly (167/300) |

By difficulty (official grader): easy 74.3% (n=74), medium 46.0% (n=126), hard 62.0% (n=50),
extra 46.0% (n=50).

## Diagnosis: overfit to a fixed, non-representative train subset

(CORRECTED 06-12 by the determinism audit: the N=100 eval is NOT a random draw — it is the
FIRST 100 of the sorted train indices, the same 100 every attempt and every run. So this is
not draw-noise winner's curse; it is overfitting to a fixed subset whose difficulty mix
differs from the full side.)

- Acceptance evals scored each attempt on the same fixed 100 of the 300 train examples.
- This strategy cleared the accuracy bar at exactly 0.68 on that subset but its skills don't
  transfer to the full distribution: train-subset 68.0 → held-out 55.7 = −12.3pp, vs the
  prior a2 strategy's train 63 → held-out 65.3 (+2.3pp).
- It is weak precisely where the held-out set is heavy: medium difficulty is 42% of the
  held-out 300 and the strategy scores 46% there.

## Fix for run #2

Evaluate every attempt on the FULL train-300 (`--eval-sample-size 300`), same bars
(0.68/0.93). Affordable because the 7B's accepted-strategy eval took only ~104s per 100
examples (~5 min per attempt at N=300). Removes draw noise from acceptance entirely.
Relaunch COLD (warm starts banned) when a GPU frees (GPU 1 = 14B baseline sweep,
GPU 2 = Spider-1.5B run #2 + GSM COLD3).

## Provenance

- Synthesis run: `outputs/generated/spider7b_cleanwin_cold_20260612/..._110258_f4631d/`
  (focal), log `spider7b_cleanwin_20260612.log`, success_report.json has the strategy.
- Held-out run: `outputs/generated/spider7b_cleanwin_heldout_20260612/..._113726_56729d/`,
  log `spider7b_cleanwin_heldout_20260612.log`.
- Rescore: `rescore_our_spider7b_heldout_20260612.py` (mirrors rescore_itergen_seed334.py;
  official execution-based Spider grader on the seed334 test-300 gold subset).
- Standing matrix entry for Spider-7B remains the accepted tie 65.3% until beaten.
