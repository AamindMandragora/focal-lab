# metaDecode iteration-cap audit

Date: 2026-07-13 IST

## Question

Could failed metaDecode synthesis cells have reached a win if synthesis had
continued beyond its configured iteration limit?

## Bottom line

**The broad hypothesis is not supported.** The evidence does support a narrower
claim: a 20-iteration cap can stop useful search too early, and moving important
runs to 40 iterations was justified. The existing evidence does **not** show
that removing the 40-iteration cap would turn most failed cells into wins.

The strongest remaining candidate for a longer cold run is **GSM Qwen3.5-9B**.
Its old 40-attempt run first reached 44.9% accuracy at attempt 32 against a
55.1% target, while already clearing the syntax target in that attempt. It was
five correct examples short on N=49. GSM Qwen3.5-4B is a weaker candidate: its
best 40-attempt result was seven correct examples short and also seven
syntax-valid examples short.

This is an inference from existing runs, not proof of what an unlimited run
would do. No run in this audit exceeded 40 planned cold synthesis iterations.

## What was inspected

- All **182** `failure_report.json` files under focal
  `/home/aadivyar/csd-generation/outputs/generated` were parsed successfully;
  parser errors: **0**.
- `results_matrix.md`, including active Qwen3.5 rows and archived Qwen2.5 rows.
- `docs/experiments/metadecode-fast-iteration-log.md`, including the H8
  iteration-budget test and the H86/H93 SMILES failure audits.
- Current Qwen3.5 resume logs and live process commands on focal.

Raw failure reports include one-step re-evaluations, infrastructure failures,
ablation arms, superseded scorers, and failed runs for cells that later won.
Those were inspected but were not treated as clean evidence about iteration
limits.

Across the 182 reports, **84** exhausted their configured attempt count. A
cleaner descriptive subset contains **38** reports with a cap of at least 20,
the configured count exhausted, and at least 75% of attempts successfully
scored. In that subset:

- 14/38 first reached their best accuracy in the final quarter of the run.
- 7/38 first reached their best accuracy in the final five attempts.

That shows that late improvements are real. It does not show that another 40 or
unlimited attempts would cross the winning bar: every member of this subset is
a failure report, and the runs differ in model, task, scorer, author, and
framework version.

## Qwen3.5 failed-cell evidence

| Cell | Existing search evidence | Distance or late trend | Iteration-cap verdict |
|---|---|---|---|
| GSM 2B | Initial cap 30: best 16.3%/98.0% at attempt 20. New cycle completed through attempt 40: best 10.2%/42.9% at attempt 28; final attempt 4.1%/42.9%. | Target 26.5%/91.8%; the longer cycle regressed. | **Against.** More attempts did not recover the gap. |
| GSM 4B | Old cap 40: best 30.6%/67.3% at attempt 32. Current newer run has reached attempt 27 with the same best accuracy, 30.6%, and improved best-paired syntax of 77.6%. | Target 44.9%/91.8% = 22/49 correct and 45/49 syntax-valid. Best paired result is 15/49 and 38/49. | **Weak/inconclusive.** A late old peak exists, but two searches have not moved accuracy past 30.6%. |
| GSM 9B | Old cap 40: best 44.9%/95.9% at attempt 32. Current newer run is around attempt 15 with best 44.9%/83.7%. | Target 55.1%/91.8% = 27/49 correct and 45/49 syntax-valid. Old best was 22/49 and 47/49. | **Best support among current failures.** It is close, the old peak was late, and syntax is already solvable. Still unproven. |
| Spider 4B | Initial cap 30: best 53.7%/97.7% at attempt 17 against 66.0%/97.0%. The later re-bar cycle completed through attempt 40; its final two scored attempts peaked at 40.3%/72.0%. | No late approach to the accuracy bar. | **Against.** The added search did not improve the original peak. |
| Spider 9B | Initial synthesis accepted on train at attempt 20, but held-out was 64.7%/99.0% versus IterGen 67.0%/98.3%. The re-bar cycle completed through attempt 40; attempts 37-40 peaked at 49.3%/95.0%. | This is primarily a train-to-held-out/generalization failure, followed by a weaker new search. | **Against as a pure cap explanation.** More training attempts do not directly fix held-out generalization. |
| SMILES 4B acrylates | Full 40 scored attempts. Best accuracy was 28%/32% at attempt 15; best syntax was 8%/100% at attempt 27; best of the last five was 4%/100%. | Target 38%/90%; accuracy and syntax were never combined and late attempts regressed. | **Strongly against.** |
| SMILES 9B isocyanates, H86/H93 failures | Two separate 40-attempt runs both topped out at 37% training accuracy, at attempts 20 and 36. The later successful campaign found an accepted strategy at attempt 4 after helper/framework changes. | The same cell was unlocked quickly only after the available operations and feedback changed. | **Against more-iterations-only.** The evidence points to search-space quality, not cap length. |

### Qwen3.5 runs that looked like failures but are not cap failures

- SMILES 2B acrylates has a 40-attempt failure report containing no scored
  attempts, followed by successful reports. This is not clean cap evidence.
- The July re-bar synthesis runs for SMILES 2B chain extenders, 4B chain
  extenders, 9B acrylates, and 9B chain extenders produced training success
  reports at attempts 18, 15, 24, and 14 respectively. Any remaining issue for
  those cells is held-out evaluation or result recording, not synthesis
  exhaustion.
- The currently running SMILES 9B isocyanates re-bar job has not failed yet and
  is not classified as a completed failure.

## Archived Qwen2.5 evidence

### Direct or near-direct 20-versus-40 comparisons

| Campaign | Shorter run | Longer run | What it shows |
|---|---|---|---|
| Spider 1.5B fast-50 H7/H8 | Cap 20 reached 58%/100% at attempt 15 and was still improving. | H8 raised the cap to 40 in a new cold run; it reached only 52%/100% at attempt 17, then stayed flat for 11 attempts before an engine crash at attempt 28. | The promising late trajectory did **not** reproduce as a win with a larger cap. Cold-run variation mattered more than the nominal cap. |
| Spider 1.5B full train-300 | A family of 20-attempt runs peaked late, commonly around attempts 13-17. | One 40-attempt run first peaked at attempt 38 (47.3%/99.0%). A separate 40-attempt cold run accepted at attempt 29 and produced a held-out win of 56.3%/99.0%. | **Supports 40 rather than 20.** It does not establish value beyond 40; the win occurred by attempt 29. |
| Spider 7B full train-300 | A 20-attempt run peaked at attempt 17 with 62.3%/96.7%. | Two 40-attempt runs peaked at attempt 31 with 69.3% and 69.7% training accuracy. Held-out results were only 62.0% and 60.3%, below IterGen 65.7%. | More iterations improved training but did **not** produce a held-out win; later search increased overfitting. |

### Other archived failure families

- **Qwen2.5 GSM:** longer 20/30-attempt failures sometimes found late accuracy
  improvements, but the retained wins appeared after tokenizer/span-closing
  fixes or in new cold runs, often by attempts 4-7. Existing evidence does not
  identify the iteration cap as the main blocker.
- **Qwen2.5 SMILES:** the 7B chain-extenders pilot stayed at 2% unique-valid
  through a 20-attempt run. After rolling-prompt parity and the
  past-complete-span helper were added, the replacement run accepted on attempt
  1. Other final SMILES wins arrived by attempts 7, 13, or 15 after framework
  changes. This is strong evidence for a missing operation or poor search space,
  not insufficient iterations.
- **Older 30/40-attempt ablations:** some first peaked late, including attempts
  32-37, which confirms that a fixed cap can truncate stochastic search. Many
  used superseded authors, scorers, model variants, or infrastructure and did
  not produce retained held-out wins. They are descriptive evidence only.

## Hypothesis verdict

### Verified

1. Moving from 20 to 40 attempts can matter. At least one retained Spider-1.5B
   held-out win came from a strategy found at attempt 29, and several training
   peaks arrived after attempt 20.
2. Forty attempts have already failed repeatedly on several cells. In multiple
   cases, late attempts were flat or worse than the earlier best.
3. For Spider 7B, the stronger 40-attempt training strategies generalized worse
   than the earlier strategy. The extra search improved training, not the final
   comparison.
4. Several failed cells later won quickly after a framework/helper change.

### Inference

- **“Twenty attempts is always enough” is false.** Forty is a safer standard
  budget for difficult cells.
- **“Uncapping beyond 40 will recover most failures” is unlikely.** The observed
  failure modes are usually semantic model limits, accuracy/syntax tradeoffs,
  train-to-held-out gaps, or missing useful operations.
- **GSM Qwen3.5-9B is the strongest exception worth testing.** Its distance to
  the bar is small, its syntax target has already been cleared, and its old best
  arrived at attempt 32. GSM Qwen3.5-4B is a secondary, weaker test case.

## Clean test of the hypothesis, if requested later

The fair test is not to continue from an existing strategy. Under the current
project rule, synthesis must start cold. A clean test would preregister one
variable—the maximum iteration budget—and compare multiple independent cold
runs under otherwise identical settings, with a 40-attempt control and a larger
fixed cap such as 80. Final judgment must use the disjoint held-out result, not
the best training attempt. No such experiment was launched during this audit.

## Reproduction

The read-only summarizer used for the inventory was copied temporarily to focal
as `/tmp/summarize_metadecode_failures.py`. Its complete output is
`/tmp/all_metadecode_failure_summary.tsv` on focal. It parsed every
`failure_report.json` under `outputs/generated`; the error log
`/tmp/all_metadecode_failure_errors.log` is empty.
