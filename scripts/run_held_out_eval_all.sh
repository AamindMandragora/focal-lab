#!/bin/bash
# Runs held-out eval for CRANE + all lottery winners, in parallel across 2 GPUs.
# Reads strategies from lottery_manifest.json.
#
# Usage:
#   bash scripts/run_held_out_eval_all.sh <gpu_a> <gpu_b>
#
# Output: outputs/generated-csd/runs/held_out_eval/{label}.json for each label.

set -e
GPU_A=${1:-1}
GPU_B=${2:-2}

cd "$(dirname "$0")/.."
SCRIPT=scripts/held_out_eval.py
PY=/opt/anaconda/bin/python3

# CRANE + winners 1-3 on GPU A, winners 4-7 on GPU B
A_STRATS=(
  "crane   /home/aadivyar/csd-generation/outputs/baselines/crane_baseline_current/GeneratedCSD.py"
  "winner1 /home/aadivyar/csd-generation/outputs/generated-csd/runs/20260422_085632_b2c7e3/synth_lottery_1/GeneratedCSD.py"
  "winner2 /home/aadivyar/csd-generation/outputs/generated-csd/runs/20260422_085632_a7eb6c/synth_lottery_2/GeneratedCSD.py"
  "winner3 /home/aadivyar/csd-generation/outputs/generated-csd/runs/20260422_090113_dd2ff8/synth_lottery_3_20260422_090702_e38577/GeneratedCSD.py"
)
B_STRATS=(
  "winner4 /home/aadivyar/csd-generation/outputs/generated-csd/runs/20260422_090915_d1c05b/synth_lottery_5_20260422_091623_5c1bcf/GeneratedCSD.py"
  "winner5 /home/aadivyar/csd-generation/outputs/generated-csd/runs/20260422_093901_2ba81f/synth_lottery_8/GeneratedCSD.py"
  "winner6 /home/aadivyar/csd-generation/outputs/generated-csd/runs/20260422_093544_4f2d16/synth_lottery_9/GeneratedCSD.py"
  "winner7 /home/aadivyar/csd-generation/outputs/generated-csd/runs/20260422_094531_23ada2/synth_lottery_10_20260422_095426_5050a1/GeneratedCSD.py"
)

run_series() {
  local gpu=$1; shift
  for entry in "$@"; do
    local label=$(echo $entry | awk '{print $1}')
    local path=$(echo $entry | awk '{print $2}')
    echo "===== [$gpu] Starting $label ====="
    $PY $SCRIPT --module "$path" --label "$label" --gpu $gpu --sample-size 50 --seed 456
    echo "===== [$gpu] Done $label ====="
  done
}

run_series $GPU_A "${A_STRATS[@]}" &
PID_A=$!
run_series $GPU_B "${B_STRATS[@]}" &
PID_B=$!

wait $PID_A $PID_B
echo "All held-out evals complete."
