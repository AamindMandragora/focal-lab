#!/bin/bash
# Phase A baselines for Qwen2.5-14B-Instruct on Spider (N=100).
# GPU 2 chain: IterGen first, then SynCode (== CRANE-degenerate per the locked
# Spider framing). Both via run_legacy_fixed_strategy. On Spider, IterGen routes
# to the legacy adapter (legacy/itergen native harness); SynCode (--strategy gcd)
# routes to run_gcd_legacy_adapter which uses vendored SynCode in grammar_strict
# mode (matches the in-house 1.5B/7B SynCode bar rows).
#
# Inputs : Qwen/Qwen2.5-14B-Instruct, Spider eval split (default)
# Outputs: outputs/baselines/{itergen,gcd}/Qwen_Qwen2.5_14B_Instruct/spider__tb1__ms600.json
# Wall   : ~30-90 min each at N=100.
set -u
cd /home/aadivyar/csd-generation
OUT_ITER=outputs/baselines/itergen/Qwen_Qwen2.5_14B_Instruct
OUT_GCD=outputs/baselines/gcd/Qwen_Qwen2.5_14B_Instruct
mkdir -p "$OUT_ITER" "$OUT_GCD"
LOG_ITER=/home/aadivyar/csd-generation/logs/baselines_14B/spider_itergen_20260601.log
LOG_GCD=/home/aadivyar/csd-generation/logs/baselines_14B/spider_syncode_20260601.log

export LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:${LD_LIBRARY_PATH:-}
PY=/apps/conda/advayth2/envs/advayth2/bin/python

# --- IterGen ---
CUDA_VISIBLE_DEVICES=2 nohup "$PY" -m synthesis.evaluate.run_legacy_fixed_strategy \
  --strategy itergen --dataset spider \
  --eval-model Qwen/Qwen2.5-14B-Instruct --eval-backend vllm \
  --eval-sample-size 100 --eval-max-steps 600 --eval-step-token-budget 1 \
  --device cuda --vllm-gpu-memory-utilization 0.85 --vllm-tensor-parallel-size 1 \
  --output-json "$OUT_ITER/spider__tb1__ms600.json" \
  > "$LOG_ITER" 2>&1
echo "[ItGn done] exit=$? log=$LOG_ITER" >> "$LOG_ITER"

# --- SynCode (CRANE-degenerate) ---
CUDA_VISIBLE_DEVICES=2 nohup "$PY" -m synthesis.evaluate.run_legacy_fixed_strategy \
  --strategy gcd --dataset spider \
  --eval-model Qwen/Qwen2.5-14B-Instruct --eval-backend vllm \
  --eval-sample-size 100 --eval-max-steps 600 --eval-step-token-budget 1 \
  --device cuda --vllm-gpu-memory-utilization 0.85 --vllm-tensor-parallel-size 1 \
  --output-json "$OUT_GCD/spider__tb1__ms600.json" \
  > "$LOG_GCD" 2>&1
echo "[GCD done] exit=$? log=$LOG_GCD" >> "$LOG_GCD"
echo "GPU2 chain done"
