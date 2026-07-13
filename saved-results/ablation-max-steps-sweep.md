# Ablation — Max Steps Sweep (Decode Budget, GSM-1.5B)

**Date:** 2026-06-12

## What this is

Redesigned decode-budget ablation requested by user. Sweeps `--eval-max-steps` from 64 up to 900 at fixed token budget tb=1, re-evaluating the same winning strategy (gsm1p5b_seed429_warmstart_body.dfy) on the same disjoint held-out 49-example split (seed429). The goal: find where the step budget actually starts to hurt, giving a finer picture than the earlier coarse ms256/512/1024 grid.

All runs are pure re-evals — no synthesis, no strategy change. The tb2/tb4 invariance result (57.1%/87.8% at ms900) was established in the prior ablation and is unchanged.

## 8-Point Max-Steps Curve

| max steps | accuracy % | syntax % |
|-----------|-----------|---------|
| 64        | 40.8%     | 81.6%   |
| 128       | 42.9%     | 63.3%   |
| 192       | 53.1%     | 81.6%   |
| 256       | 53.1%     | 83.7%   |
| 384       | 57.1%     | 87.8%   |
| 512       | 57.1%     | 87.8%   |
| 700       | 57.1%     | 87.8%   |
| 900 (ref) | 57.1%     | 87.8%   |

N = 49 held-out examples for all points. ms1024 was also measured and equals 57.1%/87.8% (confirms plateau continues above 900).

## How produced

- **Launch script:** `launch_gsm1p5b_msweep_20260611.sh` on focal GPU 1
- **Runtime:** ~3–7 minutes per point
- **Results location:** `/home/aadivyar/csd-generation/outputs/generated/gsm1p5b_seed429_ablate_tb1_ms{64,128,192,384,700,1024}_20260611/<timestamp>/results/success_report.json`
- ms256 and ms512 were already run on 2026-06-11 under the earlier coarse ablation; confirmed identical here.
- Scores read from `evaluation_result.accuracy` and `evaluation_result.syntax_rate` fields.

## Takeaway

The strategy reaches its full accuracy (57.1%) and syntax rate (87.8%) by ms384 and stays flat all the way to ms900+, so the win is not dependent on a generous step budget. Below ms256, both metrics drop — syntax collapses notably at ms128 (63.3%), suggesting some examples need 256–384 steps to close their constrained spans cleanly. The practical "safe floor" is ms384: cutting below that risks measurable accuracy and syntax loss, while anything from 384 upward is redundant.

## Commands to reproduce

```bash
# On focal, for each MS in 64 128 192 384 700:
python run_eval_only.py \
  --eval-strategy gsm1p5b_seed429_warmstart_body.dfy \
  --eval-split seed429-heldout-49 \
  --token-budget 1 \
  --eval-max-steps $MS \
  --output-tag gsm1p5b_seed429_ablate_tb1_ms${MS}_20260611
```
