# Full baseline and cold synthesis campaign launch

Date: 2026-08-03

## Purpose

Collect matched train-side baseline data for five strategies, then launch cold
MetaDecode synthesis with thresholds derived from those exact artifacts.

## Matrix

- Strategies: unconstrained, GCD, CRANE, IterGen, CARS.
- Models: Qwen 2.5 1.5B, Qwen 2.5 7B, Qwen 3.5 2B, Qwen 3.5 4B.
- Cohorts: GSM-Symbolic, Spider, and SMILES acrylates, chain extenders, and
  isocyanates.
- Baselines: 100 jobs total.
- Cold synthesis: 20 jobs, 40 author attempts each, 800 attempts total.

## Matched train settings

| Cohort | Examples | Maximum steps | Step token budget |
| --- | ---: | ---: | ---: |
| GSM-Symbolic | 49 | 900 | 1 |
| Spider | 300 | 176 | 1 |
| Each SMILES class | 50 | 400 | 1 |

GSM-Symbolic and Spider use the tracked canonical train split. Each SMILES job
uses exactly one named class.

## Threshold policy

- Accuracy: `(maximum exact correct count across all five baselines + 1) / N`.
- Perfect-baseline exception: if the maximum is `N / N`, require 95% and label
  the cell `perfect_baseline_95_percent_exception`.
- Syntax: `min(maximum exact syntax count / N, 0.90)`.
- Every threshold record includes SHA-256 hashes for all five raw baseline
  artifacts.

## Runtime and provenance

- Isolated focal worktree:
  `/home/aadivyar/csd-generation-worktrees/full-baseline-campaign-20260803`
- Branch: `codex/full-baseline-campaign-20260803`
- Baseline controller GPUs: `0,2,3`; GPU 1 is excluded.
- Claude author: `claude-opus-5` through the approved
  `/home/aadivyar/.claude-csd-synthesis` Max profile for
  `aadivya@fermi.ai`.
- Cold starts only; no initial strategy fields are included.
- Held-out evaluation is run without author API credentials.

## Reuse

The baseline command is:

```bash
python scripts/run_focal_collection_pool.py \
  --repo "$PWD" \
  --python /apps/conda/aadivyar/envs/csd/bin/python \
  --campaign full-baseline-20260803 \
  --gpu-ids 0,2,3 \
  --max-workers 8 \
  --poll-seconds 5 \
  --max-retries 1
```

The waiting builder is:

```bash
python -m scripts.runtime.build_full_baseline_cold_manifest \
  --repo "$PWD" \
  --python /apps/conda/aadivyar/envs/csd/bin/python \
  --baseline-controller-pid 3347573 \
  --gpus 0,2,3 \
  --poll-seconds 30
```

It writes the final evidence and manifest only after all 100 baseline artifacts
pass structural and count checks, validates them again, and then hands the 20
cold jobs to the existing GPU-aware cold queue.
