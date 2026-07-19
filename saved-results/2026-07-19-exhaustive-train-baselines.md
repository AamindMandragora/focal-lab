# Exhaustive main-grid train baseline measurements

Date: 2026-07-19

## Purpose

This records the same-side train comparators required for exhaustive GSM-Symbolic and Spider main-grid coverage. All measurements are local GPU runs with no paid author call. GSM uses CRANE on the `train` side of `gsm_symbolic_crane_proportional_49x49_seed123.json`. Spider uses IterGen completions on the 300 indices from the `train` side of `spider_dev_proportional_300x300_seed334.json`; the IterGen adapter's `test_indices` were checked to equal those canonical `train_indices` in the same order. Final Spider counts were replayed from each raw `completion` with this repo's official `synthesis.evaluate.benchmarks.sql_spider.executor.execute_accuracy` scorer because the raw IterGen rows' embedded gold/`exec` fields do not match that canonical train order.

## GSM-Symbolic train-49 CRANE

| Model | Correct / N | Accuracy | Strict synthesis bar | Raw artifact |
|---|---:|---:|---:|---|
| Qwen2.5-1.5B-Instruct | 8 / 49 | 0.163265 | 9 / 49 = 0.183673 | `outputs/controlled_comparison/gsm_1p5B_train/crane.json`, SHA-256 `36d7093ec553b33232197800ea259efeb52fcd1119cc5b8c9b8b612fb14ef666` |
| Qwen2.5-7B-Instruct | 18 / 49 | 0.367347 | 19 / 49 = 0.387755 | `outputs/controlled_comparison/gsm_7B_train/crane.json`, SHA-256 `76bc00ba654cab19abe421c62c9d408fc2c99d02fdeb5954217135c7fd3fa91d` |
| Qwen2.5-14B-Instruct | 20 / 49 | 0.408163 | 21 / 49 = 0.428571 | `outputs/controlled_comparison/gsm_14B_train/crane.json`, SHA-256 `6555f153e17aff738228360b7177236d5af78872e45adb5192adb968c05f5a3a` |
| Qwen3.5-2B | 7 / 49 | 0.142857 | 8 / 49 = 0.163265 | `outputs/controlled_comparison/gsm_35_2B_train/crane.json`, SHA-256 `5161788457f47aeb0a57baf7cb8b04f0598fa2d5f1f314aff828cf516780c72b` |
| Qwen3.5-4B | 16 / 49 | 0.326531 | 17 / 49 = 0.346939 | `outputs/controlled_comparison/gsm_35_4B_train/crane.json`, SHA-256 `41cf26221718ad3572542c4edddaaa9069768c9795b0e7a51d78560c1c4f85aa` |
| Qwen3.5-9B | 15 / 49 | 0.306122 | 16 / 49 = 0.326531 | `outputs/controlled_comparison/gsm_35_9B_train/crane.json`, SHA-256 `b285c146ca290f861c4ca99de1f242d842e11aeb953f575b5a04107835675544` |

## Spider train-300 IterGen

| Model | Correct / N | Accuracy | Strict synthesis bar | Raw artifact SHA-256 |
|---|---:|---:|---:|---|
| Qwen2.5-1.5B-Instruct | 160 / 300 | 0.533333 | 161 / 300 = 0.536667 | `9fc0bc147b568fa96cccebaa9f71eb77aa5a475a15a3f403c02f18e0ffd75d11` |
| Qwen2.5-7B-Instruct | 198 / 300 | 0.660000 | 199 / 300 = 0.663333 | `365009e46f0fa4861d2cfc6cc4905d970939625169b6739d00331762c6a83506` |
| Qwen2.5-14B-Instruct | 201 / 300 | 0.670000 | 202 / 300 = 0.673333 | `1a96ee94a7618a3b4bca9f7f2d99081c02c665d0a3dd53168e63e1e7e6b09c3f` |
| Qwen3.5-2B | 123 / 300 | 0.410000 | 124 / 300 = 0.413333 | `77694c5e3d441dd863af169af4bc3b2794386cd11c1c55c6f0f7907536b11484` |
| Qwen3.5-4B | 195 / 300 | 0.650000 | 196 / 300 = 0.653333 | `885431e1cfecaf5ccc4045d1ff8a85c1e749be42da1ee35a902ed7df74444afa` |
| Qwen3.5-9B | 201 / 300 | 0.670000 | 202 / 300 = 0.673333 | `23d852ce10b23fb3c49501f39fd92fa204965277a027c0808518123abd5318fd` |

The Spider raw files are under `/home/aadivyar/itergen/results/sql_results/<model>/itergen_seed334_TRAIN300_tempt:None_seed:0_rp:0.3_maxiter_20_num:300.jsonl`.

## Reproduce

GSM runs use:

```text
python -m synthesis.evaluate.run_legacy_fixed_strategy --strategy crane --dataset gsm_symbolic --eval-backend huggingface --eval-sample-size 49 --eval-max-steps 900 --eval-step-token-budget 1 --gsm-split-file environment/benchmark_splits/gsm_symbolic_crane_proportional_49x49_seed123.json --gsm-split-name train
```

Spider runs use IterGen `case_studies/sql/eval_sql_seed334_test300.py` with the adapter file whose 300 `test_indices` exactly equal the canonical split's 300 `train_indices`. Final counts are obtained by loading those canonical train examples and passing the 300 saved `completion` strings to `synthesis.evaluate.benchmarks.sql_spider.executor.execute_accuracy(..., etype="exec")`.
