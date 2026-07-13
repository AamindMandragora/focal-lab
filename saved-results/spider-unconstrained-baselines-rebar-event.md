# Spider unconstrained baselines (fixed adapter) — and the Spider-7B re-bar event

**Date:** 2026-06-11

> **SUPERSEDED BY USER RULING (same day): rows DROPPED from the matrix, re-bar reverted.**
> These runs used OUR few-shot prompt, not IterGen's zero-shot native prompt. The user ruled
> baselines must be prompted identically to the baseline repo's harness, so the 44.0%/72.3%
> numbers are NOT valid baseline rows and the Spider-7B bar is back to IterGen 65.7%.
> Why unconstrained beat IterGen here: the single inline few-shot example (worth ~7-25pp to
> Qwen Instruct models), not anything about decoding. Kept below as a record + the data
> still answers "what does the model do on our prompt without constraints" if ever needed
> as an explicitly-labeled ablation (user chose not to include it in the matrix).

## What this is

First valid measurement of the unconstrained Spider baseline on the exact seed334 test-300
split, after fixing the adapter bug (a chat-message list was passed where vLLM needs a plain
string — all earlier "unconstrained" rows showed 0.0% because of that bug plus markdown-fence
artifacts; fix details in `unconstrained-spider-prompt-bug-fix.md`).

Setup: greedy (temp 0.0), max 600 new tokens, the SAME flat few-shot prompt our metaDecode
eval uses (so the model demonstrates the ` SQL: <<...>>` output format), no grammar mask.
Scored two ways: pipeline scorer and the official Spider grader (canonical per project rules;
span content extracted from `<< >>` before grading, whitespace flattened to one line).

## Results (official grader, N=300)

| Model | Acc (official) | Syntax | Acc (pipeline) | Notes |
|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct | **44.0%** | ~98.9% (3 syntax err / 251 spans emitted) | 40.7% | easy .784 / med .349 / hard .44 / extra .16 |
| Qwen2.5-7B-Instruct | **72.3%** | 96.3% (11 syntax err / 288 spans emitted) | 47.0% | easy .838 / med .69 / hard .82 / extra .54 |

Raw outputs: `outputs/baselines/unconstrained/Qwen_Qwen2.5_{1.5B,7B}_Instruct/spider_seed334_test300_unconstrained_fixedprompt.json` on focal.

## Why this matters — the re-bar event

- **1.5B: constraints help.** Unconstrained 44.0% < metaDecode 51.0%. The CSD story holds for
  the weak model. No re-bar (the 1.5B bar is in-house IterGen 52.0%, already being chased by
  the rebar52b synthesis run).
- **7B: constraints HURT.** Unconstrained 72.3% beats metaDecode 65.3%, IterGen 65.7%, and
  CARS 61.0%. Plain few-shot prompting outperforms every constrained method at this scale,
  while the syntax gain from constraining is only ~0.7pp (96.3% → 97%). Under the user's
  re-bar rule, **the Spider-7B metaDecode target is now 72.3% accuracy** (syntax within
  10–15pp of 96.3% tolerated as a last resort).
- Contradiction to root-cause: the winning 7B strategy's own rationale says the constraint
  "barely intervenes (1/100 examples)" — yet it loses ~21 net examples to prompt-only
  generation with the same prompt and greedy decoding. A diagnosis agent is diffing the
  unconstrained-right ∧ metaDecode-wrong bucket (writeup will land at
  `spider7b-unconstrained-gap-diagnosis.md`).

## Caveats / context

- The 7B pipeline-vs-official gap (47.0 vs 72.3) is the largest seen; official is canonical
  (the same harness reproduced CARS-7B 61.0% in agreement with the pipeline the same day, so
  the harness itself is validated).
- This "unconstrained" row uses OUR few-shot prompt — it is the prompt-only ablation of our
  method, and also the honest "what does the model do without your machinery" baseline a
  reviewer would ask for. IterGen's published unconstrained row (0.0%) used a bare prompt.

## Reproduce

`bash launch_unconstrained_spider_seed334_20260611.sh` from `/home/aadivyar/csd-generation`
(GPU 1, util 0.20 for 1.5B / 0.45 for 7B). Official rescore: build task_id/completion jsonl
with `<< >>` extraction, then `RESULTS_JSONL=... python case_studies/sql/rescore_itergen_seed334.py`
from `/home/aadivyar/itergen` with the 3-path PYTHONPATH.
