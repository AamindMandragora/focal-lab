#!/usr/bin/env bash
set -euo pipefail

# H85: future paid Bedrock Qwen3.5-9B isocyanates cold SMILES matrix cell.
# Dry-run by default. Real launch is intentionally gated because H81 is the
# active paid SMILES cell and must be recorded before another paid SMILES launch.
# Scientific target: fill the next unrun live-CARS SMILES cell using the generic
# COLD UV recipe, without an initial strategy and without learned class tricks.

REPO_ROOT="${REPO_ROOT:-/home/aadivyar/csd-generation}"
cd "${REPO_ROOT}"
DRY_RUN="${DRY_RUN:-1}"
SAFE_GPU_ID="${SAFE_GPU_ID:-TO_FILL_WHEN_SAFE}"
CONFIRM_ACCOUNT="${CONFIRM_BEDROCK_ACCOUNT_887730490125:-no}"
ALLOW_WHILE_H65_RUNNING="${ALLOW_WHILE_H65_RUNNING:-no}"
MIN_FREE_MIB="${MIN_FREE_MIB:-30000}"
LOG_DIR="/tmp/csd_h85_logs"
ARTIFACT_ROOT="outputs/generated/h85_smiles_qwen35_9b_isocyanates_livebar_materialization_20260630"
PROVENANCE_DIR="${ARTIFACT_ROOT}/provenance_$(date -u +%Y%m%dT%H%M%SZ)"
PID_FILE="${LOG_DIR}/h85_smiles_qwen35_9b_isocyanates_livebar_20260630.pid"
LOG_FILE="${LOG_DIR}/h85_smiles_qwen35_9b_isocyanates_livebar_20260630.log"
CMD=(
  ./pilot_smiles_uv_qwen35_i40.sh
  Qwen/Qwen3.5-9B
  qwen35_9b
  isocyanates
  "${SAFE_GPU_ID}"
  0.55
  0.92
  0.50
)
OUT_JSON="outputs/controlled_comparison/smiles_qwen35_9b/isocyanates/metadecode_uv.json"
GEN_ROOT="outputs/generated/smiles_qwen35_9b_isocyanates_uv_qwen35_0627"
H81_PID_FILE="/tmp/csd_h81_logs/h81_smiles_qwen35_2b_acrylates_uv36_syn085_20260630.pid"
H65_PID_FILE="/tmp/csd_h65_logs/h65_gsm9_timeoutguard_20260630.pid"

if [[ "${DRY_RUN}" == "1" ]]; then
  mkdir -p "${ARTIFACT_ROOT}"
  python - <<PY_H85_DRY_RUN
import json
from pathlib import Path
payload = {
  "hypothesis": "H85",
  "dry_run": True,
  "model_calls": 0,
  "gpu_calls": 0,
  "billed_api_calls": 0,
  "account_record": "887730490125",
  "scientific_target": "cold generic Qwen3.5-9B isocyanates SMILES UV matrix cell",
  "planned_command": [
    "./pilot_smiles_uv_qwen35_i40.sh",
    "Qwen/Qwen3.5-9B",
    "qwen35_9b",
    "isocyanates",
    "${SAFE_GPU_ID}",
    "0.55",
    "0.92",
    "0.50"
  ],
  "planned_output_json": "${OUT_JSON}",
  "planned_generated_root": "${GEN_ROOT}",
  "max_iterations": 40,
  "eval_model": "Qwen/Qwen3.5-9B",
  "class": "isocyanates",
  "util": 0.55,
  "min_acc": 0.92,
  "min_syn": 0.50,
  "uses_initial_strategy_file": False,
  "requires_real_launch_env": [
    "DRY_RUN=0",
    "SAFE_GPU_ID=<gpu>",
    "CONFIRM_BEDROCK_ACCOUNT_887730490125=yes"
  ],
  "h81_guard": "real launch refuses while H81 pid is alive",
  "h65_guard": "real launch refuses while H65 pid is alive unless ALLOW_WHILE_H65_RUNNING=yes",
  "gpu_gate": "real launch requires explicit GPU, at least 30000 MiB free, and no compute process already on the chosen GPU",
  "provenance_hardening": {
    "artifact_root": "${ARTIFACT_ROOT}",
    "prelaunch_snapshot_dir": "${PROVENANCE_DIR}/prelaunch",
    "postlaunch_snapshot_dir": "${PROVENANCE_DIR}/postlaunch",
    "snapshots_current_heldout_json": "${OUT_JSON}",
    "snapshots_latest_run_metadata": "${GEN_ROOT}/latest_run.txt"
  }
}
Path("${ARTIFACT_ROOT}/h85_dry_run.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, sort_keys=True))
PY_H85_DRY_RUN
  printf 'H85 dry-run only; no paid/model/GPU/API call launched.\n'
  printf 'planned_log_file=%s\n' "${LOG_FILE}"
  printf 'planned_pid_file=%s\n' "${PID_FILE}"
  printf 'planned_provenance_dir=%s\n' "${PROVENANCE_DIR}"
  printf 'planned_command='
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

if [[ "${CONFIRM_ACCOUNT}" != "yes" ]]; then
  printf 'ERROR: set CONFIRM_BEDROCK_ACCOUNT_887730490125=yes before real H85 launch.\n' >&2
  exit 2
fi
if [[ "${SAFE_GPU_ID}" == "TO_FILL_WHEN_SAFE" ]]; then
  printf 'ERROR: set SAFE_GPU_ID to an explicit GPU index before real H85 launch.\n' >&2
  exit 3
fi
if [[ -f "${H81_PID_FILE}" ]]; then
  H81_PID="$(cat "${H81_PID_FILE}")"
  if [[ -n "${H81_PID}" ]] && ps -p "${H81_PID}" >/dev/null 2>&1; then
    printf 'ERROR: H81 paid SMILES pid=%s is still running; refusing H85 launch until H81 is recorded.\n' "${H81_PID}" >&2
    exit 4
  fi
fi
if [[ -f "${H65_PID_FILE}" && "${ALLOW_WHILE_H65_RUNNING}" != "yes" ]]; then
  H65_PID="$(cat "${H65_PID_FILE}")"
  if [[ -n "${H65_PID}" ]] && ps -p "${H65_PID}" >/dev/null 2>&1; then
    printf 'ERROR: H65 pid=%s is still running; refusing second paid launch without ALLOW_WHILE_H65_RUNNING=yes.\n' "${H65_PID}" >&2
    exit 5
  fi
fi
GPU_FREE_MIB="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', *' -v gpu="${SAFE_GPU_ID}" '$1 == gpu {print $2}')"
if [[ -z "${GPU_FREE_MIB}" ]]; then
  printf 'ERROR: GPU index %s not found.\n' "${SAFE_GPU_ID}" >&2
  exit 6
fi
if (( GPU_FREE_MIB < MIN_FREE_MIB )); then
  printf 'ERROR: GPU %s has only %s MiB free; need at least %s MiB.\n' "${SAFE_GPU_ID}" "${GPU_FREE_MIB}" "${MIN_FREE_MIB}" >&2
  exit 7
fi
GPU_UUID="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits | awk -F', *' -v gpu="${SAFE_GPU_ID}" '$1 == gpu {print $2}')"
if nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits |
  awk -F', *' -v uuid="${GPU_UUID}" '$1 == uuid {found=1} END {exit found ? 0 : 1}'; then
  printf 'ERROR: GPU %s already has a compute process; refusing to stack a 9B paid SMILES job.\n' "${SAFE_GPU_ID}" >&2
  exit 8
fi

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
python - <<PY_H85_PRELAUNCH_MANIFEST
import json
from pathlib import Path
payload = {
  "hypothesis": "H85",
  "prelaunch_snapshot_dir": "${PROVENANCE_DIR}/prelaunch",
  "postlaunch_snapshot_dir": "${PROVENANCE_DIR}/postlaunch",
  "generated_root": "${GEN_ROOT}",
  "heldout_json": "${OUT_JSON}",
  "planned_command": [
    "./pilot_smiles_uv_qwen35_i40.sh",
    "Qwen/Qwen3.5-9B",
    "qwen35_9b",
    "isocyanates",
    "${SAFE_GPU_ID}",
    "0.55",
    "0.92",
    "0.50"
  ],
}
Path("${PROVENANCE_DIR}/prelaunch/prelaunch_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY_H85_PRELAUNCH_MANIFEST
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
' h85_provenance_wrapper "${PROVENANCE_DIR}" "${OUT_JSON}" "${GEN_ROOT}" "${CMD[@]}" >"${LOG_FILE}" 2>&1 < /dev/null &
echo "$!" > "${PID_FILE}"
printf 'Launched H85 pid=%s log=%s pid_file=%s provenance_dir=%s\n' "$(cat "${PID_FILE}")" "${LOG_FILE}" "${PID_FILE}" "${PROVENANCE_DIR}"
