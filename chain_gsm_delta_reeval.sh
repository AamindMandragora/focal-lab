#!/usr/bin/env bash
# $0 delta re-eval: re-score the OLD GSM bests under the NEW z3 scorer, on the SAME
# 49-train split + SAME 90s budget that produced their recorded attempt numbers, so
# the accuracy change is the SCORER effect (+ vLLM sampling noise), not a budget change.
#   GSM-2B best_att7 recorded (old scorer, train): 46.9% acc / 85.7% syn
#   GSM-4B best_att5 recorded (old scorer, train): 44.9% acc / 71.4% syn
# Sequential on GPU3 to avoid OOM. Pure re-eval => no Bedrock author call => $0 billed.
set -uo pipefail
cd /home/aadivyar/csd-generation

echo "DELTA_CHAIN_START $(date)"

DATASET=gsm_symbolic GPU=3 SAMPLE_SIZE=49 SPLIT_NAME=train \
  EVAL_MAX_SECONDS=90 EVAL_MAX_STEPS=900 GPU_MEM_UTIL=0.40 \
  EVAL_MODEL=Qwen/Qwen3.5-2B \
  INITIAL_STRATEGY=/home/aadivyar/csd-generation/outputs/generated/synth_gsm_2b_seed123train_0627a/best_att7_strategy.dfy \
  OUTPUT_NAME=reeval_gsm_2b_att7_TRAIN_z3fix_0627 \
  bash reeval_strategy.sh

echo "DELTA_2B_DONE $(date)"

DATASET=gsm_symbolic GPU=3 SAMPLE_SIZE=49 SPLIT_NAME=train \
  EVAL_MAX_SECONDS=90 EVAL_MAX_STEPS=900 GPU_MEM_UTIL=0.40 \
  EVAL_MODEL=Qwen/Qwen3.5-4B \
  INITIAL_STRATEGY=/home/aadivyar/csd-generation/outputs/generated/synth_gsm_4b_seed123train_0627a/best_att5_strategy.dfy \
  OUTPUT_NAME=reeval_gsm_4b_att5_TRAIN_z3fix_0627 \
  bash reeval_strategy.sh

echo "DELTA_4B_DONE $(date)"
echo "DELTA_CHAIN_SENTINEL_DONE"
