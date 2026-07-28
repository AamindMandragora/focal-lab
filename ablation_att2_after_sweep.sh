#!/usr/bin/env bash
# Ablation §1: att2 (iter-5 peak) held-out re-eval. Waits for the max-steps sweep to finish
# (so GPU3 hosts only Wave1 + this, ~30GB, no 3-way OOM), then PURE RE-EVALs att2 on the
# seed123 EVAL (held-out) split at the default 900 max-steps. $0 (--max-iterations 1 +
# --initial-strategy-file => no author/Bedrock call). Compare vs att7 held-out (§2's 900 point)
# for the iter-5-vs-iter-10 held-out comparison.
set -uo pipefail
REPO=/home/aadivyar/csd-generation
cd "$REPO"
SWEEP=outputs/controlled_comparison/ablation_gsm2b_maxsteps_sweep.log

echo "ATT2_WAIT for SWEEP_ALL_DONE $(date)"
while ! grep -q "SWEEP_ALL_DONE" "$SWEEP" 2>/dev/null; do sleep 30; done
echo "ATT2_SWEEP_DONE_SEEN, launching re-eval $(date)"

DATASET=gsm_symbolic \
EVAL_MODEL=Qwen/Qwen3.5-2B \
GPU=3 \
INITIAL_STRATEGY="$REPO/outputs/generated/synth_gsm_2b_seed123train_0627a/att2_strategy.dfy" \
SAMPLE_SIZE=49 \
SPLIT_NAME=eval \
EVAL_MAX_STEPS=900 \
EVAL_MAX_SECONDS=1200 \
GPU_MEM_UTIL=0.35 \
OUTPUT_NAME="ablation_gsm2b_att2_heldout_maxsteps900" \
bash "$REPO/reeval_strategy.sh"
echo "ATT2_ALL_DONE $(date)"
