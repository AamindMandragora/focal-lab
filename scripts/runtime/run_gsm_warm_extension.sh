#!/usr/bin/env bash
# Narrow user-approved warm extension for GSM attempts 41-80.
set -euo pipefail

REPO="${REPO:-/home/aadivyar/csd-generation}"
PY="${PY:-/apps/conda/aadivyar/envs/csd/bin/python}"
CELL="${CELL:?CELL is required}"
SEED_FILE="${SEED_FILE:?SEED_FILE is required}"
HISTORY_FILE="${HISTORY_FILE:?HISTORY_FILE is required}"
OUTPUT_NAME="${OUTPUT_NAME:?OUTPUT_NAME is required}"
GPU="${GPU:?GPU is required}"
LOG="$REPO/logs/paid_synth_warm_extension_${CELL}.log"

case "$CELL" in
  gsm14b)
    MODEL=Qwen/Qwen2.5-14B-Instruct
    UTIL=0.81
    MIN_ACC=0.5918
    MIN_SYN=0.85
    ;;
  gsm-qwen35-4b)
    MODEL=Qwen/Qwen3.5-4B
    UTIL=0.45
    MIN_ACC=0.448979591837
    MIN_SYN=0.918367346939
    ;;
  gsm-qwen35-9b)
    MODEL=Qwen/Qwen3.5-9B
    UTIL=0.60
    MIN_ACC=0.551020408163
    MIN_SYN=0.918367346939
    ;;
  *)
    printf 'unknown CELL=%s\n' "$CELL" >&2
    exit 2
    ;;
esac

[[ -s "$SEED_FILE" ]] || { printf 'missing seed: %s\n' "$SEED_FILE" >&2; exit 2; }
[[ -s "$HISTORY_FILE" ]] || { printf 'missing history: %s\n' "$HISTORY_FILE" >&2; exit 2; }

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf '{"cell":"%s","billing_account":"887730490125","max_iterations":41,"initial_attempt_offset":39,"replay_attempt":40,"first_new_attempt":41,"last_new_attempt":80,"new_iterations":40,"seed_file":"%s","history_file":"%s","output_name":"%s","gpu":%s,"would_source_env":false,"would_call_bedrock":false}\n' \
    "$CELL" "$SEED_FILE" "$HISTORY_FILE" "$OUTPUT_NAME" "$GPU"
  exit 0
fi

if [[ "${CONFIRM_BEDROCK_ACCOUNT_887730490125:-}" != yes ]]; then
  printf 'billing confirmation missing for AWS account 887730490125\n' >&2
  exit 2
fi

mkdir -p "$REPO/logs" "$REPO/outputs/generated/$OUTPUT_NAME"
printf '%s [gsm-warm-worker] START cell=%s account=887730490125 region=us-east-1 gpu=%s replay=40 new=41-80\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$CELL" "$GPU" | tee -a "$LOG"

export LD_LIBRARY_PATH=/apps/conda/aadivyar/envs/csd/lib:${LD_LIBRARY_PATH:-}
export CUDA_VISIBLE_DEVICES="$GPU"
export HF_HOME=/home/aadivyar/.cache/huggingface
export TRANSFORMERS_CACHE=/home/aadivyar/.cache/huggingface
export CSD_DAILY_QUOTA_RETRY_SECONDS=3600
export CSD_DAILY_QUOTA_RETRY_JITTER_SECONDS=300
set -a
source "$REPO/.env"
set +a
cd "$REPO"

nohup "$PY" "$REPO/scripts/runtime/vllm_orphan_reaper.py" \
  >>"$REPO/logs/vllm_orphan_reaper.log" 2>&1 </dev/null &

task="$(python -c 'from synthesis.evaluate.benchmarks.gsm_symbolic.prompts import GSM_CRANE_COT_TASK; print(GSM_CRANE_COT_TASK)')"
args=(
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock
  --eval-model "$MODEL" --eval-backend vllm
  --max-iterations 41 --initial-attempt-offset 39
  --initial-strategy-file "$SEED_FILE"
  --initial-attempt-history-file "$HISTORY_FILE"
  --output-name "$OUTPUT_NAME" --output-dir "outputs/generated/$OUTPUT_NAME"
  --min-accuracy "$MIN_ACC" --min-syntax-rate "$MIN_SYN"
  --eval-sample-size 49 --eval-max-steps 900 --eval-step-token-budget 1
  --eval-max-seconds-per-example 600 --eval-min-examples-before-threshold-stop 49
  --max-tokens 32768 --restart-after-stuck-iters 0
  --vllm-gpu-memory-utilization "$UTIL" --vllm-max-model-len 16384 --device auto
  --adaptive-helper-mask --helper-selection-policy bandit --refinement-beam-size 2
  --anthropic-thinking enabled --anthropic-effort high --anthropic-thinking-display summarized
  --vllm-tensor-parallel-size 1
  --dataset gsm_symbolic
  --gsm-split-file "$REPO/environment/benchmark_splits/gsm_symbolic_crane_proportional_49x49_seed123.json"
  --gsm-split-name train
)

set +e
"$PY" -m synthesis.run_synthesis --task "$task" "${args[@]}" 2>&1 | tee -a "$LOG"
status=${PIPESTATUS[0]}
set -e
printf '%s [gsm-warm-worker] FINISH cell=%s exit=%s output=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$CELL" "$status" "$OUTPUT_NAME" | tee -a "$LOG"
exit "$status"
