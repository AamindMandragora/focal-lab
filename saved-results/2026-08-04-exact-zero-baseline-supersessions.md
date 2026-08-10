# Exact-zero baseline supersessions

Date: 2026-08-04

## Purpose

Preserve every quarantined artifact while recording the separate versioned
replacement that may supersede it for corrected baseline evidence.

## Accepted replacement

- Label: smiles-acrylates-qwen25-1p5b-gcd
- Quarantined SHA-256:
  f0f378de0e02c6d4120154b92531e9fec70d0b6f2d94d88ff8a71f835a0ab587
- Failure: 50 nonblank outputs collapsed to one repeated malformed answer.
- Repair commit: 951ef778f57524b8896a2920264b00e23388de97
- Repair: use temperature 0.7 sampling only for SMILES GCD; GSM and Spider
  remain greedy.
- Replacement root:
  outputs/baselines/exact-zero-repair-20260804-gcd-sampling-v2
- Pool PID: 594872
- Worker PID at launch: 594893
- GPU: 3
- Launched: 2026-08-04T22:41:55Z
- Completed: 2026-08-04T22:58:15Z; exit 0
- Replacement SHA-256:
  5287e4c61cba6e09bc2836557b17d0a4ffb64b444040d16a38aa22d0c6c6d256
- Result: accuracy 0.0; syntax rate 0.0; 50/50 complete nonblank rows;
  18 distinct outputs; 49/50 outputs reached the 400-token cap.
- Independent review: PASS by `gpt-5.6-sol`, bound to the replacement SHA.
- Status: accepted genuine zero. The decoder functioned, but its generations
  were severely repetitive and mostly failed to terminate. The predecessor
  remains preserved as invalid system-failure evidence.

## Reproduce

From the focal worktree:

    /apps/conda/aadivyar/envs/csd/bin/python -m pytest -q tests/test_gcd_smiles_sampling.py
    /apps/conda/aadivyar/envs/csd/bin/python scripts/run_focal_collection_pool.py --repo /home/aadivyar/csd-generation-worktrees/full-baseline-campaign-20260803 --python /apps/conda/aadivyar/envs/csd/bin/python --campaign full-baseline-20260803 --campaign-output-name exact-zero-repair-20260804-gcd-sampling-v2 --gpu-ids 3 --max-workers 1 --max-retries 1 --poll-seconds 5 --include-label smiles-acrylates-qwen25-1p5b-gcd

