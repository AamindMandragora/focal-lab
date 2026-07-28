#!/usr/bin/env bash
set -euo pipefail

# H63 is a paid Bedrock SMILES cell. It dry-runs by default.
# To launch for real, set:
#   DRY_RUN=0
#   SAFE_GPU_ID=<gpu_index>
#   CONFIRM_BEDROCK_ACCOUNT_887730490125=yes

REPO_ROOT="${REPO_ROOT:-/home/aadivyar/csd-generation}"
cd "${REPO_ROOT}"

DRY_RUN="${DRY_RUN:-1}"
SAFE_GPU_ID="${SAFE_GPU_ID:-TO_FILL_WHEN_SAFE}"
MIN_FREE_MIB="${MIN_FREE_MIB:-20000}"
CONFIRM_BEDROCK_ACCOUNT_887730490125="${CONFIRM_BEDROCK_ACCOUNT_887730490125:-}"

LOG_DIR="/tmp/csd_h63_logs"
PID_FILE="${LOG_DIR}/h63_smiles_qwen35_4b_isocyanates_20260629.pid"
LOG_FILE="${LOG_DIR}/h63_smiles_qwen35_4b_isocyanates_20260629.log"
OUTPUT_ROOT="outputs/generated/smiles_qwen35_4b_isocyanates_uv_qwen35_0627"
HELDOUT_JSON="outputs/controlled_comparison/smiles_qwen35_4b/isocyanates/metadecode_uv.json"

CMD=(
  ./pilot_smiles_uv_qwen35_i40.sh
  Qwen/Qwen3.5-4B
  qwen35_4b
  isocyanates
  "${SAFE_GPU_ID}"
  0.40
  0.16
  0.50
)

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'H63 dry run only; no GPU/model/API call launched.\n'
  printf 'planned_output_root=%s\n' "${OUTPUT_ROOT}"
  printf 'planned_heldout_json=%s\n' "${HELDOUT_JSON}"
  printf 'planned_pid_file=%s\n' "${PID_FILE}"
  printf 'planned_log_file=%s\n' "${LOG_FILE}"
  printf 'planned_command='
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

if [[ "${SAFE_GPU_ID}" == "TO_FILL_WHEN_SAFE" ]]; then
  printf 'ERROR: set SAFE_GPU_ID to an explicit GPU index before launching H63.\n' >&2
  exit 2
fi

if [[ "${CONFIRM_BEDROCK_ACCOUNT_887730490125}" != "yes" ]]; then
  printf 'ERROR: set CONFIRM_BEDROCK_ACCOUNT_887730490125=yes before launching this paid Bedrock run.\n' >&2
  exit 3
fi

GPU_FREE_MIB="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', *' -v gpu="${SAFE_GPU_ID}" '$1 == gpu {print $2}')"
if [[ -z "${GPU_FREE_MIB}" ]]; then
  printf 'ERROR: GPU index %s not found.\n' "${SAFE_GPU_ID}" >&2
  exit 4
fi
if (( GPU_FREE_MIB < MIN_FREE_MIB )); then
  printf 'ERROR: GPU %s has only %s MiB free; need at least %s MiB.\n' "${SAFE_GPU_ID}" "${GPU_FREE_MIB}" "${MIN_FREE_MIB}" >&2
  exit 5
fi

GPU_UUID="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits | awk -F', *' -v gpu="${SAFE_GPU_ID}" '$1 == gpu {print $2}')"
while IFS=',' read -r app_gpu_uuid app_pid _app_name _used_mem; do
  app_gpu_uuid="$(printf '%s' "${app_gpu_uuid}" | xargs)"
  app_pid="$(printf '%s' "${app_pid}" | xargs)"
  [[ -z "${app_gpu_uuid}" || "${app_gpu_uuid}" != "${GPU_UUID}" ]] && continue
  app_user="$(ps -o user= -p "${app_pid}" | xargs || true)"
  if [[ -n "${app_user}" && "${app_user}" != "aadivyar" ]]; then
    printf 'ERROR: GPU %s has non-aadivyar process pid=%s user=%s.\n' "${SAFE_GPU_ID}" "${app_pid}" "${app_user}" >&2
    exit 6
  fi
done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits || true)

mkdir -p "${LOG_DIR}"
setsid "${CMD[@]}" >"${LOG_FILE}" 2>&1 < /dev/null &
echo "$!" > "${PID_FILE}"
printf 'Launched H63 pid=%s log=%s pid_file=%s\n' "$(cat "${PID_FILE}")" "${LOG_FILE}" "${PID_FILE}"
