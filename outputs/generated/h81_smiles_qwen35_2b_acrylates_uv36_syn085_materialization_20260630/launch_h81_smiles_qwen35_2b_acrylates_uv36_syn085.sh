#!/usr/bin/env bash
set -euo pipefail

# H81: future paid Bedrock Qwen3.5-2B acrylates retry with stricter validity.
# Dry-run by default. Real launch is intentionally gated because H65 may already
# be using the approved paid Bedrock lane.
# Scientific single variable vs H70: keep min_acc at the live focal CARS UV bar 0.36,
# and raise min_syn from 0.50 to 0.85 to force valid molecules while preserving UV pressure.
# H81 provenance hardening: real launch snapshots the old acrylates held-out
# and latest-run state, then copies post-launch train/held-out artifacts into
# the H81 materialization folder.

REPO_ROOT="${REPO_ROOT:-/home/aadivyar/csd-generation}"
cd "${REPO_ROOT}"
DRY_RUN="${DRY_RUN:-1}"
SAFE_GPU_ID="${SAFE_GPU_ID:-TO_FILL_WHEN_SAFE}"
CONFIRM_ACCOUNT="${CONFIRM_BEDROCK_ACCOUNT_887730490125:-no}"
ALLOW_WHILE_H65_RUNNING="${ALLOW_WHILE_H65_RUNNING:-no}"
MIN_FREE_MIB="${MIN_FREE_MIB:-12000}"
LOG_DIR="/tmp/csd_h81_logs"
ARTIFACT_ROOT="outputs/generated/h81_smiles_qwen35_2b_acrylates_uv36_syn085_materialization_20260630"
PROVENANCE_DIR="${ARTIFACT_ROOT}/provenance_$(date -u +%Y%m%dT%H%M%SZ)"
PID_FILE="${LOG_DIR}/h81_smiles_qwen35_2b_acrylates_uv36_syn085_20260630.pid"
LOG_FILE="${LOG_DIR}/h81_smiles_qwen35_2b_acrylates_uv36_syn085_20260630.log"
CMD=(
  ./pilot_smiles_uv_qwen35_i40.sh
  Qwen/Qwen3.5-2B
  qwen35_2b
  acrylates
  "${SAFE_GPU_ID}"
  0.20
  0.36
  0.85
)
OUT_JSON="outputs/controlled_comparison/smiles_qwen35_2b/acrylates/metadecode_uv.json"
GEN_ROOT="outputs/generated/smiles_qwen35_2b_acrylates_uv_qwen35_0627"

if [[ "${DRY_RUN}" == "1" ]]; then
  python - <<PY_H81_DRY_RUN
import json
from pathlib import Path
payload = {
  "hypothesis": "H81",
  "dry_run": True,
  "model_calls": 0,
  "gpu_calls": 0,
  "billed_api_calls": 0,
  "account_record": "887730490125",
  "scientific_single_variable": "min_syn raised from H70 0.50 to 0.85 while min_acc stays at live CARS UV bar 0.36",
  "planned_command": ${CMD[@]@Q},
  "planned_output_json": "${OUT_JSON}",
  "planned_generated_root": "${GEN_ROOT}",
  "max_iterations": 40,
  "eval_model": "Qwen/Qwen3.5-2B",
  "class": "acrylates",
  "util": 0.20,
  "min_acc": 0.36,
  "min_syn": 0.85,
  "requires_real_launch_env": ["DRY_RUN=0", "SAFE_GPU_ID=<gpu>", "CONFIRM_BEDROCK_ACCOUNT_887730490125=yes"],
  "h65_guard": "real launch refuses while H65 pid is alive unless ALLOW_WHILE_H65_RUNNING=yes",
  "provenance_hardening": {
    "artifact_root": "${ARTIFACT_ROOT}",
    "prelaunch_snapshot_dir": "${PROVENANCE_DIR}/prelaunch",
    "postlaunch_snapshot_dir": "${PROVENANCE_DIR}/postlaunch",
    "snapshots_current_heldout_json": "${OUT_JSON}",
    "snapshots_latest_run_metadata": "${GEN_ROOT}/latest_run.txt"
  },
}
Path("outputs/generated/h81_smiles_qwen35_2b_acrylates_uv36_syn085_materialization_20260630/h81_dry_run.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, sort_keys=True))
PY_H81_DRY_RUN
  printf 'H81 dry-run only; no paid/model/GPU/API call launched.\n'
  printf 'planned_log_file=%s\n' "${LOG_FILE}"
  printf 'planned_pid_file=%s\n' "${PID_FILE}"
  printf 'planned_provenance_dir=%s\n' "${PROVENANCE_DIR}"
  printf 'planned_command='
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

if [[ "${CONFIRM_ACCOUNT}" != "yes" ]]; then
  printf 'ERROR: set CONFIRM_BEDROCK_ACCOUNT_887730490125=yes before real H81 launch.\n' >&2
  exit 2
fi
if [[ "${SAFE_GPU_ID}" == "TO_FILL_WHEN_SAFE" ]]; then
  printf 'ERROR: set SAFE_GPU_ID to an explicit GPU index before real H81 launch.\n' >&2
  exit 3
fi
if [[ -f /tmp/csd_h65_logs/h65_gsm9_timeoutguard_20260630.pid && "${ALLOW_WHILE_H65_RUNNING}" != "yes" ]]; then
  H65_PID="$(cat /tmp/csd_h65_logs/h65_gsm9_timeoutguard_20260630.pid)"
  if ps -p "${H65_PID}" >/dev/null 2>&1; then
    printf 'ERROR: H65 pid=%s is still running; refusing second paid launch without ALLOW_WHILE_H65_RUNNING=yes.\n' "${H65_PID}" >&2
    exit 4
  fi
fi
GPU_FREE_MIB="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', *' -v gpu="${SAFE_GPU_ID}" '$1 == gpu {print $2}')"
if [[ -z "${GPU_FREE_MIB}" ]]; then
  printf 'ERROR: GPU index %s not found.\n' "${SAFE_GPU_ID}" >&2
  exit 5
fi
if (( GPU_FREE_MIB < MIN_FREE_MIB )); then
  printf 'ERROR: GPU %s has only %s MiB free; need at least %s MiB.\n' "${SAFE_GPU_ID}" "${GPU_FREE_MIB}" "${MIN_FREE_MIB}" >&2
  exit 6
fi
GPU_UUID="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits | awk -F', *' -v gpu="${SAFE_GPU_ID}" '$1 == gpu {print $2}')"
while IFS=',' read -r app_gpu_uuid app_pid _app_name _used_mem; do
  app_gpu_uuid="$(printf '%s' "${app_gpu_uuid}" | xargs)"
  app_pid="$(printf '%s' "${app_pid}" | xargs)"
  [[ -z "${app_gpu_uuid}" || "${app_gpu_uuid}" != "${GPU_UUID}" ]] && continue
  app_user="$(ps -o user= -p "${app_pid}" | xargs || true)"
  if [[ -n "${app_user}" && "${app_user}" != "aadivyar" ]]; then
    printf 'ERROR: GPU %s has non-aadivyar process pid=%s user=%s.\n' "${SAFE_GPU_ID}" "${app_pid}" "${app_user}" >&2
    exit 7
  fi
done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits || true)

mkdir -p "${LOG_DIR}" "${PROVENANCE_DIR}/prelaunch" "${PROVENANCE_DIR}/postlaunch"
if [[ -f "${OUT_JSON}" ]]; then
  cp "${OUT_JSON}" "${PROVENANCE_DIR}/prelaunch/metadecode_uv.prelaunch.json"
fi
if [[ -f "${GEN_ROOT}/latest_run.txt" ]]; then
  cp "${GEN_ROOT}/latest_run.txt" "${PROVENANCE_DIR}/prelaunch/latest_run.prelaunch.txt"
fi
if [[ -e "${GEN_ROOT}/latest" ]]; then
  readlink "${GEN_ROOT}/latest" > "${PROVENANCE_DIR}/prelaunch/latest_symlink_target.prelaunch.txt" 2>/dev/null || true
fi
python - <<PY_H81_PRELAUNCH_MANIFEST
import json
from pathlib import Path
payload = {
  "hypothesis": "H81",
  "prelaunch_snapshot_dir": "${PROVENANCE_DIR}/prelaunch",
  "postlaunch_snapshot_dir": "${PROVENANCE_DIR}/postlaunch",
  "generated_root": "${GEN_ROOT}",
  "heldout_json": "${OUT_JSON}",
  "planned_command": ${CMD[@]@Q},
}
Path("${PROVENANCE_DIR}/prelaunch/prelaunch_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY_H81_PRELAUNCH_MANIFEST
setsid bash -c '
set +e
provenance_dir="$1"
out_json="$2"
gen_root="$3"
shift 3
"$@"
status=$?
mkdir -p "${provenance_dir}/postlaunch"
if [[ -f "${out_json}" ]]; then
  cp "${out_json}" "${provenance_dir}/postlaunch/metadecode_uv.postlaunch.json"
fi
if [[ -f "${gen_root}/latest_run.txt" ]]; then
  cp "${gen_root}/latest_run.txt" "${provenance_dir}/postlaunch/latest_run.postlaunch.txt"
  latest_run="$(cat "${gen_root}/latest_run.txt")"
  if [[ -f "${latest_run}/results/success_report.json" ]]; then
    cp "${latest_run}/results/success_report.json" "${provenance_dir}/postlaunch/train_success_report.postlaunch.json"
  fi
  if [[ -f "${latest_run}/results/failure_report.json" ]]; then
    cp "${latest_run}/results/failure_report.json" "${provenance_dir}/postlaunch/train_failure_report.postlaunch.json"
  fi
fi
if [[ -e "${gen_root}/latest" ]]; then
  readlink "${gen_root}/latest" > "${provenance_dir}/postlaunch/latest_symlink_target.postlaunch.txt" 2>/dev/null || true
fi
printf "%s\n" "${status}" > "${provenance_dir}/postlaunch/exit_status.txt"
exit "${status}"
' h81_provenance_wrapper "${PROVENANCE_DIR}" "${OUT_JSON}" "${GEN_ROOT}" "${CMD[@]}" >"${LOG_FILE}" 2>&1 < /dev/null &
echo "$!" > "${PID_FILE}"
printf 'Launched H81 pid=%s log=%s pid_file=%s provenance_dir=%s\n' "$(cat "${PID_FILE}")" "${LOG_FILE}" "${PID_FILE}" "${PROVENANCE_DIR}"
