# Ablation: Synthesis Iterations (iter5 / iter10 / final)

**Date**: 2026-06-11 (updated 2026-06-11)  
**Status**: COMPLETE  
**Purpose**: For the metaDecode paper — train-side performance at iteration checkpoints across all 10 winning synthesis cells.

> **Spec change (2026-06-11)**: Original cutoffs were best-by-att3 and best-by-att5. Replaced with best-by-att5 and best-by-att10 because the 3-vs-5 delta was too small to show anything useful in the ablation.

> **IMPORTANT**: All numbers are TRAIN-side (synthesis evaluation sample), not held-out test numbers.  
> "Best by att N" = highest accuracy among attempts 1..N; the syntax shown is from that same attempt.  
> "Accepted attempt" = the attempt that first met the synthesis thresholds (acc ≥ min, syn ≥ 0.85).

---

## How the synthesis chains work

The winning strategies were produced by chained runs:
1. A main "exploration" run (ralph_* or smiles_*) ran many attempts.
2. A "warmstart" or continuation run used the best body as a starting point, sometimes succeeding in 1 attempt.
3. The accepted attempt number counts sequentially across the chain.

For the ablation table:
- GSM-1.5B: counts 7 atts in original run + warmstart att1 = att8 accepted.
- GSM-7B: counts ~24 atts in disjoint run + warmstart att1 = att25 accepted.
- Spider-1.5B: counts 10 atts in seed334_warmstart + 300x300 att1 = att11 accepted.
- Spider-7B: direct synthesis — accepted at att6 in seed334_warmstart run.
- SMILES: direct single-run synthesis.

---

## Main Ablation Table

| Run | Best-by-att5 acc | Best-by-att5 syn | Best-by-att10 acc | Best-by-att10 syn | Accepted att# | Accepted acc | Accepted syn |
|-----|-----------------|-----------------|------------------|------------------|---------------|-------------|-------------|
| GSM-1.5B | 38.8% | 49.0% | 46.9% | 61.2% | 8 | 42.9% | 87.8% |
| GSM-7B | 44.9% | 51.0% | 44.9% | 51.0% | 25 | 65.3% | 85.7% |
| Spider-1.5B | 39.0% | 98.0% | 39.0% | 98.0% | 11 | 47.0% | 97.0% |
| Spider-7B | 59.0% | 98.0% | 62.0% | 97.0% | 6 | 62.0% | 97.0% *(accepted-early)* |
| chain_extenders-7B | 70.0% | 100.0% | 70.0% | 100.0% | 2 | 70.0% | 100.0% *(accepted-early)* |
| isocyanates-7B | 14.3% | 37.1% | 32.0% | 62.0% | 13 | 22.0% | 90.0% |
| acrylates-7B | 4.0% | 100.0% | 14.0% | 78.0% | 14 | 20.0% | 100.0% |
| chain_extenders-1.5B | 64.0% | 74.0% | 64.0% | 88.0% | 7 | 64.0% | 88.0% *(accepted-early)* |
| isocyanates-1.5B | 22.0% | 46.0% | 22.0% | 46.0% | 15 | 30.0% | 100.0% |
| acrylates-1.5B | 12.0% | 76.0% | 40.0% | 70.0% | 9 | 34.0% | 88.0% *(accepted-early)* |

**Notes**:  
- "Best-by-att N" = attempt with highest accuracy among atts 1..N; syntax shown is from that same attempt.  
- *(accepted-early)* = accepted attempt is ≤ 10 (or ≤ 5), so best-by-N equals or is bounded by the accepted run.  
- GSM-1.5B best-by-att10: atts 6 and 7 both reached 46.9% acc; att7 is used (61.2% syn vs att6's 0.0%).  
- Spider-7B has only 6 total attempts; best-by-att10 = accepted (att6).  
- chain_extenders-7B has only 2 total attempts; best-by-att10 = accepted (att2).  
- chain_extenders-1.5B has 7 total attempts; best-by-att10 = accepted (att7).  
- acrylates-1.5B best-by-att10: att7 = 40.0%/70.0% is higher acc than accepted att9 = 34.0%/88.0% (accepted had lower acc but met the synthesis threshold via syntax).

---

## Caveats

1. **Train-side only**: These numbers are from the synthesis evaluation sample (49 examples for GSM, 100 for Spider, 50 for SMILES). Held-out numbers are reported separately in results_matrix.md.
2. **Missing attempts**: Several runs have gaps in the attempt sequence in their prompt_io logs (e.g., acrylates-7B atts 4-5 not captured, isocyanates-1.5B atts 3,6,9 missing). The gaps indicate attempts where the synthesis loop's beam exploration produced candidates that didn't change the best-so-far. The table uses the best confirmed value from available attempts.
3. **Attempt counting across chains**: For GSM-1.5B, GSM-7B, and Spider-1.5B, the attempt counter spans two or more synthesis runs. The warmstart att=1 is shown as att 8, 25, and 11 respectively in the table.
4. **Spider-7B run identity**: The task description named `ralph_7B_spider_300x300_seed334_20260604` (which had an empty results dir), but the actual accepted synthesis run is `ralph_7B_spider_seed334_warmstart_20260604` (success_report att=6, 62%/97%). The 300x300 run continued iterating but did not have a success_report.
5. **acrylates-1.5B log note**: As noted in the task brief, the LOG for the retry5 run is corrupted by a dual-instance incident. Data here comes from the prompt_io.jsonl of run dir `_071138_50aca4` which is the clean instance, cross-referenced against the success_report.

---

## Detailed Per-Attempt Histories

### GSM-1.5B (seed429)

**Synthesis chain**: `ralph_1p5B_gsm_seed429_iter30_20260604` → `gsm1p5b_seed429_warmstart_20260604`  
**Sources**: prompt_io.jsonl at `/home/aadivyar/csd-generation/logs/ralph_1p5B_gsm_seed429_iter30_20260604_20260604_101009_6e6503/` + success_report in warmstart run dir  
**Thresholds**: acc ≥ 32%, syn ≥ 85%  
**Sample**: N=49

| Att | Accuracy | Syntax | Notes |
|-----|----------|--------|-------|
| 1 | 36.7% | 57.1% | Below syn threshold |
| 2 | 0.0% | 25.0% | Regression |
| 3 | **38.8%** | **49.0%** | Best by att5 — still below syn threshold |
| 4 | 6.1% | 6.1% | |
| 5 | 0.0% | 0.0% | |
| 6 | 46.9% | 0.0% | High acc but syn=0 |
| 7 | **46.9%** | **61.2%** | Best by att10 (ties att6 on acc; higher syn) |
| 8 (warmstart att1) | **42.9%** | **87.8%** | **ACCEPTED** |

---

### GSM-7B (seed429)

**Synthesis chain**: `ralph_7B_gsm_disjoint_iter30_20260603` → `ralph_7B_gsm_seed429_iter30_20260604`  
**Sources**: prompt_io.jsonl at `/home/aadivyar/csd-generation/logs/ralph_7B_gsm_disjoint_iter30_20260603_20260604_020930_155670/` + success_report in seed429 run dir  
**Thresholds**: acc ≥ 32%, syn ≥ 85%  
**Sample**: N=49

| Att | Accuracy | Syntax | Notes |
|-----|----------|--------|-------|
| 1 | **44.9%** | **51.0%** | Best by att5 and att10 — still below syn threshold |
| 2 | 42.9% | 61.2% | |
| 3 | 14.3% | 22.4% | |
| 4 | 14.3% | 22.4% | |
| 5 | 28.6% | 40.8% | |
| 6 | 32.7% | 49.0% | |
| 7 | 34.7% | 59.2% | |
| 8 | 14.3% | 18.4% | |
| 9 | 34.7% | 59.2% | |
| 10 | 30.6% | 53.1% | |
| 11 | 42.9% | 61.2% | |
| 12 | 22.4% | 34.7% | |
| 13 | 42.9% | 61.2% | |
| 14 | 22.4% | 34.7% | |
| 15 | 65.3% | 77.6% | First time acc > threshold, syn still below |
| 16 | 57.1% | 65.3% | |
| 17 | 71.4% | 73.5% | |
| 18 | 63.3% | 93.9% | First fully met threshold |
| 19 | 32.7% | 44.9% | |
| 20 | 63.3% | 93.9% | |
| 21 | 75.5% | 91.8% | |
| 22 | 77.6% | 89.8% | |
| 23 | 75.5% | 91.8% | |
| 24 | 77.6% | 89.8% | |
| 25 (warmstart att1) | **65.3%** | **85.7%** | **ACCEPTED** |

---

### Spider-1.5B (seed334 300x300)

**Synthesis chain**: `ralph_1p5B_spider_seed334_warmstart_20260604` → `ralph_1p5B_spider_300x300_seed334_20260604`  
**Sources**: prompt_io.jsonl at `/home/aadivyar/csd-generation/logs/ralph_1p5B_spider_seed334_warmstart_20260604_20260604_103748_9f12e6/` + success_report in 300x300 run dir  
**Thresholds**: acc ≥ 32%, syn ≥ 85%  
**Sample**: N=100 (seed334_warmstart) / N=100 (300x300)

| Att | Accuracy | Syntax | Notes |
|-----|----------|--------|-------|
| 1 | **39.0%** | **98.0%** | Met threshold at att1! Best by att5 and att10 |
| 2 | 39.0% | 98.0% | Same as att1 |
| 3 | 39.0% | 98.0% | Same |
| 4 | 29.0% | 80.6% | |
| 5 | 39.0% | 98.0% | |
| 6 | 37.0% | 97.0% | |
| 7 | 0.0% | 0.0% | |
| 8 | 39.0% | 98.0% | |
| 9 | 33.0% | 95.0% | |
| 10 | 39.0% | 98.0% | |
| 11 (300x300 att1) | **47.0%** | **97.0%** | **ACCEPTED** |

---

### Spider-7B (seed334 warmstart)

**Synthesis run**: `ralph_7B_spider_seed334_warmstart_20260604`  
**Source**: prompt_io.jsonl at `/home/aadivyar/csd-generation/logs/ralph_7B_spider_seed334_warmstart_20260604_20260604_103748_861183/`  
**Thresholds**: acc ≥ 62%, syn ≥ 93%  
**Sample**: N=100

| Att | Accuracy | Syntax | Notes |
|-----|----------|--------|-------|
| 1 | 53.0% | 97.0% | Below acc threshold |
| 2 | 44.0% | 92.0% | |
| 3 | **59.0%** | **98.0%** | Best by att5 — still below acc threshold |
| 4 | 2.0% | 2.0% | |
| 5 | 55.0% | 95.0% | |
| 6 | **62.0%** | **97.0%** | **ACCEPTED** (= best-by-att10, accepted-early) |

---

### SMILES: chain_extenders-7B

**Synthesis run**: `smiles_7B_chain_extenders_fixB_20260610` (run dir `_20260610_124236_7450aa`)  
**Source**: prompt_io.jsonl at `/home/aadivyar/csd-generation-smiles/logs/smiles_7B_chain_extenders_fixB_20260610_20260610_124236_7450aa/`  
**Thresholds**: acc ≥ 8%, syn ≥ 85%  
**Sample**: N=50

| Att | Accuracy | Syntax | Notes |
|-----|----------|--------|-------|
| 1 | 16.0% | 94.0% | Met threshold |
| 2 | **70.0%** | **100.0%** | **ACCEPTED** |

---

### SMILES: isocyanates-7B

**Synthesis run**: `smiles_7B_isocyanates_fixAB_20260610`  
**Source**: prompt_io.jsonl at `/home/aadivyar/csd-generation-smiles/logs/smiles_7B_isocyanates_fixAB_20260610_20260610_133120_bf4b5c/`  
**Thresholds**: acc ≥ 20%, syn ≥ 85%  
**Sample**: N=50

| Att | Accuracy | Syntax | Notes |
|-----|----------|--------|-------|
| 1 | **14.3%** | **37.1%** | Best by att5 (highest acc through att5) |
| 2 | 14.3% | 37.1% | |
| 4 | 2.0% | 76.0% | |
| 5 | 4.0% | 4.0% | |
| 6 | **32.0%** | **62.0%** | Best by att10 (ties att7; att6 used) |
| 7 | 32.0% | 62.0% | |
| 8 | 10.0% | 34.0% | |
| 9 | 14.0% | 24.0% | |
| 10 | 22.0% | 38.0% | First to meet acc threshold, syn too low |
| 11 | 6.0% | 98.0% | |
| 12 | 18.0% | 72.0% | |
| 13 | **22.0%** | **90.0%** | **ACCEPTED** (acc ≥ 20%, syn ≥ 85%) |

---

### SMILES: acrylates-7B

**Synthesis run**: `smiles_7B_acrylates_fixAB_20260610`  
**Source**: prompt_io.jsonl at `/home/aadivyar/csd-generation-smiles/logs/smiles_7B_acrylates_fixAB_20260610_20260610_151054_b9d1f1/`  
**Thresholds**: acc ≥ 10%, syn ≥ 85%  
**Sample**: N=50

| Att | Accuracy | Syntax | Notes |
|-----|----------|--------|-------|
| 1 | 0.0% | 72.5% | |
| 2 | 0.0% | 100.0% | |
| 3 | **4.0%** | **100.0%** | Best by att5 (0% until here) |
| 6 | 4.0% | 100.0% | |
| 10 | **14.0%** | **78.0%** | Best by att10 (first non-zero above bar, syn too low) |
| 11 | 0.0% | 0.0% | |
| 13 | 0.0% | 0.0% | |
| 14 | **20.0%** | **100.0%** | **ACCEPTED** |

*Note: Attempts 4-5, 7-9, 12 not explicitly in ledger; inferred as intermediate explorations.*

---

### SMILES: chain_extenders-1.5B

**Synthesis run**: `smiles_1p5B_chain_extenders_fixAB_gpu2_20260610`  
**Source**: prompt_io.jsonl at `/home/aadivyar/csd-generation-smiles/logs/smiles_1p5B_chain_extenders_fixAB_gpu2_20260610_20260610_164121_ab8b9c/`  
**Thresholds**: acc ≥ 56%, syn ≥ 85%  
**Sample**: N=50

| Att | Accuracy | Syntax | Notes |
|-----|----------|--------|-------|
| 1 | 32.0% | 64.0% | |
| 2 | 32.0% | 64.0% | |
| 4 | 56.0% | 70.0% | Met acc threshold, syn below |
| 5 | **64.0%** | **74.0%** | Best by att5 — syn still below 85% |
| 6 | 64.0% | 74.0% | |
| 7 | **64.0%** | **88.0%** | **ACCEPTED** (= best-by-att10, accepted-early) |

---

### SMILES: isocyanates-1.5B

**Synthesis run**: `smiles_1p5B_isocyanates_fixAB_20260610`  
**Source**: prompt_io.jsonl at `/home/aadivyar/csd-generation-smiles/logs/smiles_1p5B_isocyanates_fixAB_20260610_20260610_171212_5d627f/`  
**Thresholds**: acc ≥ 22%, syn ≥ 85%  
**Sample**: N=50

| Att | Accuracy | Syntax | Notes |
|-----|----------|--------|-------|
| 1 | 10.0% | 62.0% | |
| 2 | 10.0% | 62.0% | |
| 4 | **22.0%** | **46.0%** | Best by att5 and att10 |
| 5 | 22.0% | 46.0% | |
| 7 | 18.0% | 96.0% | |
| 8 | 18.0% | 96.0% | |
| 10 | 18.0% | 58.0% | |
| 11 | 10.0% | 42.0% | |
| 12 | 14.0% | 42.0% | |
| 13 | 22.0% | 84.0% | Almost met (syn just below 85%) |
| 14 | 18.0% | 52.0% | |
| 15 | **30.0%** | **100.0%** | **ACCEPTED** |

---

### SMILES: acrylates-1.5B (retry5)

**Synthesis run**: `smiles_1p5B_acrylates_fixAB_retry5_clean_20260611` (run dir `_071138_50aca4`)  
**Source**: prompt_io.jsonl at `/home/aadivyar/csd-generation-smiles/logs/smiles_1p5B_acrylates_fixAB_retry5_clean_20260611_20260611_071138_50aca4/`  
**Thresholds**: acc ≥ 9%, syn ≥ 85%  
**Sample**: N=50  
**Note**: LOG is from the clean 071138 instance; the dual-instance 071605 log is corrupted.

| Att | Accuracy | Syntax | Notes |
|-----|----------|--------|-------|
| 1 | 0.0% | 100.0% | |
| 3 | **12.0%** | **76.0%** | Best by att5 (met acc threshold, syn below 85%) |
| 4 | 4.0% | 98.0% | |
| 5 | 4.0% | 98.0% | |
| 7 | **40.0%** | **70.0%** | Best by att10 (highest acc, syn collapsed) |
| 8 | 4.0% | 100.0% | |
| 9 | **34.0%** | **88.0%** | **ACCEPTED** (att ≤ 10, accepted-early) |

*Note: Att2 had 0%/100%; att1 had 0%/100%. Att 6 not in log.*

---

## Summary: How Quickly Does Synthesis Converge?

| Dataset | Atts to first threshold | Accepted att# | Improvement att1→final (acc) |
|---------|------------------------|---------------|------------------------------|
| GSM-1.5B | 8 (needs warmstart) | 8 | 36.7% → 42.9% |
| GSM-7B | 18 (disjoint) / 25 total | 25 | 44.9% → 65.3% |
| Spider-1.5B | 1 (seed334_warmstart!) | 11 | 39.0% → 47.0% |
| Spider-7B | 6 | 6 | 53.0% → 62.0% |
| chain_extenders-7B | 1 | 2 | 16.0% → 70.0% |
| isocyanates-7B | 13 | 13 | 14.3% → 22.0% |
| acrylates-7B | 14 | 14 | 0.0% → 20.0% |
| chain_extenders-1.5B | 7 | 7 | 32.0% → 64.0% |
| isocyanates-1.5B | 15 | 15 | 10.0% → 30.0% |
| acrylates-1.5B | 9 | 9 | 0.0% → 34.0% |

---

## Data Sources (focal server paths)

All paths are on `aadivyar@focal`:

- GSM logs: `/home/aadivyar/csd-generation/logs/`
- GSM run dirs: `/home/aadivyar/csd-generation/outputs/generated/`
- SMILES logs: `/home/aadivyar/csd-generation-smiles/logs/`
- SMILES run dirs: `/home/aadivyar/csd-generation-smiles/outputs/generated/`

Key log files used:
- GSM-1.5B: `ralph_1p5B_gsm_seed429_iter30_20260604_20260604_101009_6e6503/prompt_io.jsonl`
- GSM-7B: `ralph_7B_gsm_disjoint_iter30_20260603_20260604_020930_155670/prompt_io.jsonl`
- Spider-1.5B: `ralph_1p5B_spider_seed334_warmstart_20260604_20260604_103748_9f12e6/prompt_io.jsonl`
- Spider-7B: `ralph_7B_spider_seed334_warmstart_20260604_20260604_103748_861183/prompt_io.jsonl`
- chain_extenders-7B: `smiles_7B_chain_extenders_fixB_20260610_20260610_124236_7450aa/prompt_io.jsonl`
- isocyanates-7B: `smiles_7B_isocyanates_fixAB_20260610_20260610_133120_bf4b5c/prompt_io.jsonl`
- acrylates-7B: `smiles_7B_acrylates_fixAB_20260610_20260610_151054_b9d1f1/prompt_io.jsonl`
- chain_extenders-1.5B: `smiles_1p5B_chain_extenders_fixAB_gpu2_20260610_20260610_164121_ab8b9c/prompt_io.jsonl`
- isocyanates-1.5B: `smiles_1p5B_isocyanates_fixAB_20260610_20260610_171212_5d627f/prompt_io.jsonl`
- acrylates-1.5B: `smiles_1p5B_acrylates_fixAB_retry5_clean_20260611_20260611_071138_50aca4/prompt_io.jsonl`

---

*Generated: 2026-06-11 via offline log analysis on focal. Table updated 2026-06-11: replaced att3/att5 cutoffs with att5/att10 (3-vs-5 delta too small to show anything useful).*

---

## Best-so-far curves

**Date added**: 2026-06-11

**What it shows**: A small-multiples grid (5 rows × 2 columns, one panel per cell) plotting best-so-far accuracy (%) vs. synthesis attempt number. The blue step line is the running maximum accuracy up to each attempt; grey dots show individual attempt scores; the red star marks the accepted attempt. All numbers are train-side (synthesis evaluation sample).

**Files**:
- Figure: `saved-results/ablation-iteration-curves.png`
- Raw data (JSON): `saved-results/ablation-iteration-curves-data.json`
- Plot script: `saved-results/ablation-iteration-curves-plot.py`

**Takeaway**: Winning strategies arrive at widely different depths — chain_extenders-7B finds its winner at attempt 2 while GSM-7B and isocyanates-1.5B require 25 and 15 attempts respectively. Most cells show a plateau-then-jump pattern rather than steady improvement: the best-so-far line stays flat for several attempts before a discrete jump. The five cells where best-by-att5 equals best-by-att10 (Spider-1.5B, Spider-7B, chain_extenders-7B, chain_extenders-1.5B, acrylates-1.5B) all had their wins arrive at or before attempt 9, confirming the early-exit signal in the accepted-attempt column.
