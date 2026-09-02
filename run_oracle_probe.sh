#!/bin/bash
# Self-consistency ORACLE CEILING probe (k=10, T=1.0).
# Runs the SAME Spider-7B strategy on a ~35-example structural-failure split:
#   - 1 control run at CSD_CONSTRAINED_TEMPERATURE=0.0  (must == argmax answers)
#   - 10 sample runs at CSD_CONSTRAINED_TEMPERATURE=1.0
# Goal: measure the ORACLE recovery ceiling = fraction of examples where >=1 of
# the 10 samples matches gold by execution. If that ceiling can't reach ~+8pp,
# no sampling-based selection method can close the gap to IterGen 65.7%.
# Sequential on GPU 1 (free). ~1.5-3h total (11 runs x 35 examples).
set -u

export LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:${LD_LIBRARY_PATH:-}
export CUDA_VISIBLE_DEVICES=1
cd /home/aadivyar/csd-generation

PY=/apps/conda/advayth2/envs/advayth2/bin/python
STRATEGY=/home/aadivyar/csd-generation/spider7b_300x300_a2_63pct_body.dfy
SPLIT=/home/aadivyar/csd-generation/environment/benchmark_splits/spider_oracle_struct_seed334.json
BASE=outputs/generated/sc_oracle_probe

run_one () {
  local name="$1"; local temp="$2"
  export CSD_CONSTRAINED_TEMPERATURE="$temp"
  local out="$BASE/$name"
  mkdir -p "$out"
  echo "=== $name  (T=$temp) START ==="
  $PY -m synthesis.run_synthesis \
    --task 'Generate a single valid SQL query as exactly `SQL: <<YOUR QUERY>>`, using only the provided schema context.' \
    --dataset spider \
    --max-iterations 1 \
    --initial-strategy-file "$STRATEGY" \
    --generation-model us.anthropic.claude-sonnet-4-6 \
    --generation-backend bedrock \
    --eval-model "Qwen/Qwen2.5-7B-Instruct" \
    --eval-backend vllm \
    --output-name "$name" \
    --min-accuracy 0.0 \
    --min-syntax-rate 0.0 \
    --eval-sample-size 35 \
    --eval-max-steps 1200 \
    --eval-step-token-budget 1 \
    --eval-max-seconds-per-example 180 \
    --eval-min-examples-before-threshold-stop 35 \
    --vllm-gpu-memory-utilization 0.50 \
    --device auto \
    --output-dir "$out" \
    --adaptive-helper-mask \
    --helper-selection-policy bandit \
    --anthropic-thinking enabled \
    --anthropic-effort high \
    --anthropic-thinking-display summarized \
    --vllm-tensor-parallel-size 1 \
    --spider-split-file "$SPLIT" \
    --spider-split-name test \
    > "/tmp/sc_oracle_${name}.log" 2>&1
  echo "=== $name DONE (exit $?) ==="
}

run_one oracle_ctrl_t00 0.0
for i in $(seq 1 10); do
  run_one "oracle_s${i}_t10" 1.0
done
echo "ALL ORACLE PROBE RUNS COMPLETE"
