# Spider seed334 Difficulty Imbalance Analysis

**Date:** 2026-06-13  
**Question:** Is the seed334 train/test split difficulty-imbalanced? Is the 62.3% train → 57.7% held-out drop for att17 a difficulty-mix artifact or genuine overfitting?

---

## Split Construction

The split was built with `split_strategy: stratified_proportional`. Train-300 and test-300 have identical difficulty counts — no imbalance.

| Band | Train N | Test N |
|------|---------|--------|
| easy | 72 | 72 |
| medium | 129 | 129 |
| hard | 51 | 51 |
| extra | 48 | 48 |
| **Total** | **300** | **300** |

The two sets do not overlap (0 shared examples).

---

## Per-Band Accuracy for Att17

| Band | Train N | Train acc | Test N | Test acc | Gap |
|------|---------|-----------|--------|----------|-----|
| easy | 72 | 84.7% | 72 | 83.3% | −1.4 pp |
| medium | 129 | 51.2% | 129 | 45.7% | −5.5 pp |
| hard | 51 | 70.6% | 51 | 56.9% | −13.7 pp |
| extra | 48 | 50.0% | 48 | 52.1% | +2.1 pp |
| **TOTAL** | **300** | **62.3%** | **300** | **57.7%** | **−4.6 pp** |

---

## Cross-Strategy Comparison on Test-300

| Strategy | Easy | Medium | Hard | Extra | Total |
|----------|------|--------|------|-------|-------|
| att17 | 83.3% | 45.7% | 56.9% | 52.1% | 57.7% |
| cleanwin | 76.4% | 45.7% | 60.8% | 45.8% | 55.7% |
| 63pct warmstart | 79.2% | 53.5% | 51.0% | 41.7% | 57.3% |

---

## Conclusion

**Verdict: Genuine overfitting in the hard and medium bands. Not a difficulty-mix artifact.**

- Re-weighting would produce the same number — the split is perfectly balanced.
- Easy holds flat (−1.4 pp), extra is noise (+2.1 pp on N=48).
- Medium drops 5.5 pp on stable N=129 — real but modest.
- Hard drops 13.7 pp on N=51 — main driver of the overall gap.
- The hard-band N (51 training examples) is too small for the strategy to learn a general rule; it learned schema-specific heuristics that don't transfer to the 51 different hard test queries.
- Medium is stuck at 45–53% across all three strategies, suggesting a ceiling on what the current synthesis approach can extract from medium-difficulty queries.

**Where to look next:** Inspect what att17 does on hard-band training queries vs. hard-band test queries — specifically whether it relies on schema-specific patterns (table/column names) that happen to appear in the 51 training hard examples but not in the test ones.
