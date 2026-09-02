#!/usr/bin/env bash
# Ablation §2 (max-steps / token-budget sweep) for GSM-2B, plus §1's att7 held-out point.
# PURE RE-EVAL of the iter-10 peak strategy (best_att7_strategy.dfy) on the seed123 EVAL
# (held-out) split at four EVAL_MAX_STEPS budgets. All $0: --max-iterations 1 +
# --initial-strategy-file => NO author/Bedrock call (see reeval_strategy.sh header).
# Co-resident on GPU 3 with Wave 1 at 0.35 mem-util (~14GB into ~24GB free). Sequential
# so only one extra vLLM is ever loaded alongside Wave 1.
set -uo pipefail

REPO=/home/aadivyar/csd-generation
cd "$REPO"

STRAT="$REPO/outputs/generated/synth_gsm_2b_seed123train_0627a/best_att7_strategy.dfy"
GPU=3
MEM=0.35

echo "SWEEP_START gsm2b att7 held-out max-steps sweep gpu=$GPU $(date)"
for MS in 256 512 900 1024; do
  echo "SWEEP_STEP max_steps=$MS $(date)"
  DATASET=gsm_symbolic \
  EVAL_MODEL=Qwen/Qwen3.5-2B \
  GPU=$GPU \
  INITIAL_STRATEGY="$STRAT" \
  SAMPLE_SIZE=49 \
  SPLIT_NAME=eval \
  EVAL_MAX_STEPS=$MS \
  EVAL_MAX_SECONDS=1200 \
  GPU_MEM_UTIL=$MEM \
  OUTPUT_NAME="ablation_gsm2b_att7_heldout_maxsteps${MS}" \
  bash "$REPO/reeval_strategy.sh"
  echo "SWEEP_STEP_DONE max_steps=$MS $(date)"
done
echo "SWEEP_ALL_DONE $(date)"
