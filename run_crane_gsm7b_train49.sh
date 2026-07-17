#!/bin/bash
# CRANE baseline on the seed123 TRAIN-49 (same 49 as the june att4 arm-B re-eval),
# for the B-loses/CRANE-wins per-example comparison. Mirrors gsm7b_baselines_rerun.sh
# (which produced the eval-49 controlled files) with split-name train.
set -uo pipefail
cd /home/aadivyar/csd-generation
export CUDA_VISIBLE_DEVICES=3
export LD_LIBRARY_PATH=/opt/anaconda/lib:${LD_LIBRARY_PATH:-}
mkdir -p outputs/controlled_comparison/gsm_7B_train
python -m synthesis.evaluate.run_legacy_fixed_strategy \
  --strategy crane --dataset gsm_symbolic --eval-model Qwen/Qwen2.5-7B-Instruct \
  --eval-backend huggingface \
  --eval-sample-size 49 --eval-max-steps 900 --eval-step-token-budget 1 \
  --gsm-split-file environment/benchmark_splits/gsm_symbolic_crane_proportional_49x49_seed123.json \
  --gsm-split-name train \
  --output-json outputs/controlled_comparison/gsm_7B_train/crane.json
echo "EXIT_crane_gsm7B_train=$?"
echo DONE_CRANE_TRAIN49
