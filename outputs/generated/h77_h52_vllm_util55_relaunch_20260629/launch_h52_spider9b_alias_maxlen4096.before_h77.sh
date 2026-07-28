#!/usr/bin/env bash
set -euo pipefail

# H52 Spider-9B alias-postprocessor held-out retry.
#
# Inputs:
#   SAFE_GPU_ID: GPU index that has at least 30 GiB free and no non-aadivyar process.
#   DRY_RUN=1: print the command without starting the process.
#
# Outputs:
#   /tmp/csd_h52_logs/h52_spider9b_alias_maxlen4096_20260629.{pid,log}
#   outputs/generated/h52_spider9b_alias_postprocess_heldout_maxlen4096_20260629/h52_reeval.json
#
# Algorithm:
#   1. require the caller to choose a safe GPU explicitly;
#   2. for a real launch, re-check free memory and process ownership on that GPU;
#   3. strip paid-provider credential variables from the child environment;
#   4. run the already-compiled H51 Spider strategy through the local vLLM re-evaluator;
#   5. write the background PID for monitoring.

if [[ -z "${SAFE_GPU_ID:-}" ]]; then
  echo "SAFE_GPU_ID is required; choose a GPU only after verifying >=30 GiB free and no non-aadivyar process." >&2
  exit 2
fi

if [[ ! "${SAFE_GPU_ID}" =~ ^[0-9]+$ ]]; then
  echo "SAFE_GPU_ID must be a numeric GPU index, got: ${SAFE_GPU_ID}" >&2
  exit 2
fi

WORKTREE="/home/aadivyar/.config/superpowers/worktrees/csd-generation/h51-spider-alias-postprocess"
COMPILED="${WORKTREE}/tmp_friction_scratch/template_verify/_build_pt-py/GeneratedCSD.py"
OUT_DIR="/home/aadivyar/csd-generation/outputs/generated/h52_spider9b_alias_postprocess_heldout_maxlen4096_20260629"
OUT_JSON="${OUT_DIR}/h52_reeval.json"
LOG_DIR="/tmp/csd_h52_logs"
LOG_PATH="${LOG_DIR}/h52_spider9b_alias_maxlen4096_20260629.log"
PID_PATH="${LOG_DIR}/h52_spider9b_alias_maxlen4096_20260629.pid"
SPIDER_SPLIT="/home/aadivyar/csd-generation/environment/benchmark_splits/spider_dev_proportional_300x300_seed334.json"
PYTHON="/apps/conda/aadivyar/envs/csd/bin/python"

if [[ ! -f "${COMPILED}" ]]; then
  echo "Missing compiled strategy: ${COMPILED}" >&2
  exit 3
fi

if [[ ! -f "${SPIDER_SPLIT}" ]]; then
  echo "Missing Spider split: ${SPIDER_SPLIT}" >&2
  exit 3
fi

if [[ "${DRY_RUN:-0}" != "1" ]]; then
  gpu_csv="$(nvidia-smi -i "${SAFE_GPU_ID}" --query-gpu=memory.used,memory.total --format=csv,noheader,nounits)"
  used_mib="$(echo "${gpu_csv}" | awk -F, '{gsub(/ /, "", $1); print $1}')"
  total_mib="$(echo "${gpu_csv}" | awk -F, '{gsub(/ /, "", $2); print $2}')"
  free_mib="$((total_mib - used_mib))"
  if (( free_mib < 30000 )); then
    echo "GPU ${SAFE_GPU_ID} has only ${free_mib} MiB free; need at least 30000 MiB." >&2
    exit 4
  fi

  while read -r existing_pid; do
    [[ -z "${existing_pid}" ]] && continue
    owner="$(ps -o user= -p "${existing_pid}" | awk '{print $1}')"
    if [[ -n "${owner}" && "${owner}" != "aadivyar" ]]; then
      echo "GPU ${SAFE_GPU_ID} has non-aadivyar process ${existing_pid} owned by ${owner}; refusing launch." >&2
      exit 5
    fi
  done < <(nvidia-smi -i "${SAFE_GPU_ID}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)
fi

mkdir -p "${OUT_DIR}" "${LOG_DIR}"
cd "${WORKTREE}"

CMD=(
  env
  -u AWS_BEARER_TOKEN_BEDROCK
  -u AWS_ACCESS_KEY_ID
  -u AWS_SECRET_ACCESS_KEY
  -u AWS_SESSION_TOKEN
  -u AWS_PROFILE
  -u OPENAI_API_KEY
  -u ANTHROPIC_API_KEY
  CUDA_VISIBLE_DEVICES="${SAFE_GPU_ID}"
  VLLM_WORKER_MULTIPROC_METHOD=spawn
  "${PYTHON}"
  -m synthesis.scripts.reevaluate_compiled_csd
  "${COMPILED}"
  --dataset spider
  --eval-model Qwen/Qwen3.5-9B
  --eval-backend vllm
  --device auto
  --sample-size 300
  --max-steps 200
  --max-seconds-per-example 600
  --step-token-budget 1
  --vllm-gpu-memory-utilization 0.50
  --vllm-tensor-parallel-size 1
  --vllm-max-model-len 4096
  --spider-split-file "${SPIDER_SPLIT}"
  --spider-split-name eval
  --output-json "${OUT_JSON}"
)

printf 'H52 command:'
printf ' %q' "${CMD[@]}"
printf '\n'

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1; not launching."
  exit 0
fi

nohup "${CMD[@]}" >"${LOG_PATH}" 2>&1 &
echo "$!" >"${PID_PATH}"
echo "Launched H52 PID $(cat "${PID_PATH}")"
echo "Log: ${LOG_PATH}"
echo "Output JSON: ${OUT_JSON}"
