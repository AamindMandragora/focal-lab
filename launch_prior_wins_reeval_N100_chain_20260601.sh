#!/bin/bash
# Re-evaluate the 4 prior metaDecode WINS at N=100 to validate them
# apples-to-apples vs the N=100 baselines they beat. Sequential on GPU 2,
# starts after the Spider-14B reeval (already on GPU 2) finishes.
#
# Inputs : 4 GeneratedCSD.py files (winning attempts per the matrix / fallback_winner.json)
# Outputs: /home/aadivyar/csd-generation/outputs/reeval/<short>_N100.json each + per-run log
# Spider regex fix: confirmed live at synthesis/evaluate/evaluator.py:1689 (.*? DOTALL).
set -u
cd /home/aadivyar/csd-generation
mkdir -p logs/reeval outputs/reeval
export LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:${LD_LIBRARY_PATH:-}
PY=/apps/conda/advayth2/envs/advayth2/bin/python

GSM_SPLIT=/home/aadivyar/csd-generation/environment/benchmark_splits/gsm_symbolic_crane_proportional.json

GSM_15B=/home/aadivyar/csd-generation/outputs/generated/ralph_1p5B_gsm_relaunch_20260530_fixedlib/ralph_1p5B_gsm_relaunch_20260530_fixedlib_20260530_155808_ebc2d8/python/ralph_1p5B_gsm_relaunch_20260530_fixedlib_20260530_164045_a98c11/GeneratedCSD.py
GSM_7B=/home/aadivyar/csd-generation/outputs/generated/ralph_7B_gsm_instruct_win_20260601_v2/ralph_7B_gsm_instruct_win_20260601_v2_20260601_082228_83dcde/python/ralph_7B_gsm_instruct_win_20260601_v2_20260601_085715_9846d5/GeneratedCSD.py
SPI_15B=/home/aadivyar/csd-generation/outputs/generated/ralph_1p5B_spider_win_20260531/ralph_1p5B_spider_win_20260531_20260531_182250_e49fe6/python/ralph_1p5B_spider_win_20260531_20260531_191655_0c6a71/GeneratedCSD.py
SPI_7B=/home/aadivyar/csd-generation/outputs/generated/ralph_7B_spider_win_20260531/ralph_7B_spider_win_20260531_20260531_182250_18f184/python/ralph_7B_spider_win_20260531_20260531_184045_2ca6e8/GeneratedCSD.py

CHAIN_LOG=logs/reeval/prior_wins_reeval_chain_20260601.log
echo "[chain $(date)] waiting for Spider-14B reeval (PID 2637075) to finish before starting" > "$CHAIN_LOG"
# Wait for the Spider-14B reeval to free GPU 2
while pgrep -f "reevaluate_compiled_csd .*ralph_14B_spider_instruct_win_20260601" >/dev/null 2>&1; do
  sleep 30
done
sleep 20  # let vLLM tear down cleanly
echo "[chain $(date)] GPU 2 free, starting 4-step re-eval chain" >> "$CHAIN_LOG"

run_one() {
  local short="$1"; local compiled="$2"; local dataset="$3"; local model="$4"
  local maxsteps="$5"; local memutil="$6"; local extra="$7"
  local out="outputs/reeval/${short}_N100.json"
  local log="logs/reeval/${short}_N100_20260601.log"
  echo "[chain $(date)] start $short -> $out" >> "$CHAIN_LOG"
  CUDA_VISIBLE_DEVICES=2 "$PY" -m synthesis.scripts.reevaluate_compiled_csd \
    "$compiled" \
    --dataset "$dataset" \
    --eval-model "$model" --eval-backend vllm \
    --sample-size 100 --max-steps "$maxsteps" --step-token-budget 1 \
    --vllm-gpu-memory-utilization "$memutil" --vllm-tensor-parallel-size 1 \
    --max-seconds-per-example 180 \
    $extra \
    --output-json "$out" \
    > "$log" 2>&1
  echo "[chain $(date)] done  $short (exit=$?)" >> "$CHAIN_LOG"
  sleep 15  # vLLM teardown breathing room
}

# Smallest first so we see results soonest, then 7B, then GSM
run_one "ralph_1p5B_spider_win_20260531"            "$SPI_15B" spider       "Qwen/Qwen2.5-1.5B-Instruct" 600 0.45 ""
run_one "ralph_7B_spider_win_20260531"              "$SPI_7B"  spider       "Qwen/Qwen2.5-7B-Instruct"   600 0.50 ""
run_one "ralph_1p5B_gsm_relaunch_20260530_fixedlib" "$GSM_15B" gsm_symbolic "Qwen/Qwen2.5-1.5B-Instruct" 900 0.45 "--gsm-split-file $GSM_SPLIT --gsm-split-name eval"
run_one "ralph_7B_gsm_instruct_win_20260601_v2"     "$GSM_7B"  gsm_symbolic "Qwen/Qwen2.5-7B-Instruct"   900 0.50 "--gsm-split-file $GSM_SPLIT --gsm-split-name eval"

echo "[chain $(date)] all 4 reevals complete" >> "$CHAIN_LOG"
