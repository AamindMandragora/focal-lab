#!/usr/bin/env bash
# Ready-to-fire COLD GSM-2B relaunch with the VERIFIED CRANE-2B bar (0.245/0.837).
# Replaces the dead canary (which used 0.0/0.0 -> accepted attempt 1 and quit).
# Usage:  bash fire_gsm2b.sh <GPU_INDEX>
# Billed: Sonnet-4.6 Bedrock author, AWS 887730490125 (UIUC focal lab, pre-approved).
set -uo pipefail
GPU="${1:?usage: bash fire_gsm2b.sh <GPU_INDEX>}"
REPO=/home/aadivyar/csd-generation
NAME=synth_gsm_2b_seed123train_z3fix_0627b
OUT="$REPO/outputs/generated/$NAME"
mkdir -p "$OUT"
# Record the exact launch command this time (the dead canary had none).
cat > "$OUT/launch_cmd.txt" <<EOF
DATASET=gsm_symbolic EVAL_MODEL=Qwen/Qwen3.5-2B GPU=$GPU MIN_ACC=0.245 MIN_SYN=0.837 \
MAX_ITERS=40 SAMPLE_SIZE=49 SPLIT_NAME=train OUTPUT_NAME=$NAME bash run_synth_cell.sh
bar = verified CRANE-2B held-out (24.5% acc / 83.7%* syn; *syn unverified, gate only)
launched $(date)
EOF
cd "$REPO"
DATASET=gsm_symbolic EVAL_MODEL=Qwen/Qwen3.5-2B GPU="$GPU" \
  MIN_ACC=0.245 MIN_SYN=0.837 MAX_ITERS=40 SAMPLE_SIZE=49 \
  SPLIT_NAME=train OUTPUT_NAME="$NAME" \
  nohup bash run_synth_cell.sh > "$OUT/launch.log" 2>&1 &
echo "FIRED GSM-2B relaunch on GPU $GPU, pid $!, out $OUT"
