#!/bin/bash
# Decode-budget ablations (2026-06-11) on the GSM-1.5B WINNING strategy (seed429 warmstart
# body; reference held-out 57.1%/87.8% at tb1/ms900, reproduced bit-exact 3x).
# Grid: max-steps ms in {64,128,192,384,700} at tb1 (fills in gradual curve; 256/512/900/1024
# already measured). Pure re-evals (max-iterations 1, bars 0) on the SAME held-out 49 split —
# only the budget knobs vary. GPU 1 (22.6 GB free as of launch).
set -u

export LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:${LD_LIBRARY_PATH:-}
cd /home/aadivyar/csd-generation

STRATEGY=/home/aadivyar/csd-generation/gsm1p5b_seed429_warmstart_body.dfy
GSM_SPLIT=/home/aadivyar/csd-generation/environment/benchmark_splits/gsm_symbolic_crane_proportional_49x49_seed429.json

run_one () {
  local TB=$1 MS=$2
  local NAME="gsm1p5b_seed429_ablate_tb${TB}_ms${MS}_20260611"
  local OUT="outputs/generated/$NAME"
  mkdir -p "$OUT"
  echo "=== ABLATION tb=$TB ms=$MS $(date) ==="
  CUDA_VISIBLE_DEVICES=1 /apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
    --task 'Solve math word problems step by step, wrapping intermediate symbolic expressions and the final answer inside << >> delimiters.' \
    --dataset gsm_symbolic \
    --max-iterations 1 \
    --initial-strategy-file "$STRATEGY" \
    --generation-model us.anthropic.claude-sonnet-4-6 \
    --generation-backend bedrock \
    --eval-model "Qwen/Qwen2.5-1.5B-Instruct" \
    --eval-backend vllm \
    --output-name "$NAME" \
    --min-accuracy 0.0 \
    --min-syntax-rate 0.0 \
    --eval-sample-size 49 \
    --eval-max-steps "$MS" \
    --eval-step-token-budget "$TB" \
    --eval-max-seconds-per-example 90 \
    --eval-min-examples-before-threshold-stop 49 \
    --vllm-gpu-memory-utilization 0.40 \
    --device auto \
    --output-dir "$OUT" \
    --adaptive-helper-mask \
    --helper-selection-policy bandit \
    --anthropic-thinking enabled \
    --anthropic-effort high \
    --anthropic-thinking-display summarized \
    --vllm-tensor-parallel-size 1 \
    --gsm-split-file "$GSM_SPLIT" \
    --gsm-split-name eval
  echo "EXIT_ABLATE_tb${TB}_ms${MS}=$?"
}

# reference is tb1/ms900 = 57.1/87.8 (already recorded; not re-run)
# curve fill-ins: 256/512/900/1024 already measured; adding 64/128/192/384/700
run_one 1 64
run_one 1 128
run_one 1 192
run_one 1 384
run_one 1 700
echo "DONE_GSM1P5B_MSWEEP $(date)"
