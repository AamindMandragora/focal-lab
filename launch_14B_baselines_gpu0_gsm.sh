#!/bin/bash
# Phase A baselines for Qwen2.5-14B-Instruct on GSM-Symbolic (N=100).
# GPU 0 chain: CRANE first, then IterGen. Both via run_legacy_fixed_strategy,
# which routes GSM strategies through the upstream CRANE repo subprocess
# (the canonical paper-faithful pipeline used for the in-house 1.5B/7B bars).
#
# Inputs : Qwen/Qwen2.5-14B-Instruct, gsm_symbolic_crane_proportional eval split
# Outputs: outputs/baselines/{crane,itergen}/Qwen_Qwen2.5_14B_Instruct/gsm_symbolic__tb1__ms900.json
# Wall   : first load downloads ~28 GB; each baseline run ~30-90 min at N=100.
set -u
cd /home/aadivyar/csd-generation
SPLIT=/home/aadivyar/csd-generation/environment/benchmark_splits/gsm_symbolic_crane_proportional.json
OUT_CRANE=outputs/baselines/crane/Qwen_Qwen2.5_14B_Instruct
OUT_ITER=outputs/baselines/itergen/Qwen_Qwen2.5_14B_Instruct
mkdir -p "$OUT_CRANE" "$OUT_ITER"
LOG_CRANE=/tmp/baseline_14B_gsm_crane_20260601.log
LOG_ITER=/tmp/baseline_14B_gsm_itergen_20260601.log

export LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:${LD_LIBRARY_PATH:-}
PY=/apps/conda/advayth2/envs/advayth2/bin/python

# --- CRANE ---
CUDA_VISIBLE_DEVICES=0 nohup "$PY" -m synthesis.evaluate.run_legacy_fixed_strategy \
  --strategy crane --dataset gsm_symbolic \
  --eval-model Qwen/Qwen2.5-14B-Instruct --eval-backend vllm \
  --eval-sample-size 100 --eval-max-steps 900 --eval-step-token-budget 1 \
  --device cuda --vllm-gpu-memory-utilization 0.85 --vllm-tensor-parallel-size 1 \
  --gsm-split-file "$SPLIT" --gsm-split-name eval \
  --output-json "$OUT_CRANE/gsm_symbolic__tb1__ms900.json" \
  > "$LOG_CRANE" 2>&1
echo "[CRANE done] exit=$? log=$LOG_CRANE" >> "$LOG_CRANE"

# --- IterGen (after CRANE finishes) ---
CUDA_VISIBLE_DEVICES=0 nohup "$PY" -m synthesis.evaluate.run_legacy_fixed_strategy \
  --strategy itergen --dataset gsm_symbolic \
  --eval-model Qwen/Qwen2.5-14B-Instruct --eval-backend vllm \
  --eval-sample-size 100 --eval-max-steps 900 --eval-step-token-budget 1 \
  --device cuda --vllm-gpu-memory-utilization 0.85 --vllm-tensor-parallel-size 1 \
  --gsm-split-file "$SPLIT" --gsm-split-name eval \
  --output-json "$OUT_ITER/gsm_symbolic__tb1__ms900.json" \
  > "$LOG_ITER" 2>&1
echo "[ItGn done] exit=$? log=$LOG_ITER" >> "$LOG_ITER"
echo "GPU0 chain done"
