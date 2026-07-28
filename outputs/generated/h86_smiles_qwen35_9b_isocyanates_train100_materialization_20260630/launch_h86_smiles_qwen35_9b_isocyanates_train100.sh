#!/usr/bin/env bash
set -euo pipefail

# H86: paid Bedrock Qwen3.5-9B isocyanates SMILES run with train/held-out
# sample-size alignment. Dry-run by default. This supersedes launching H85
# as-is after H81 showed the 50-example train UV gate can be too weak for the
# 100-example held-out UV metric.

REPO_ROOT="${REPO_ROOT:-/home/aadivyar/csd-generation}"
cd "${REPO_ROOT}"
DRY_RUN="${DRY_RUN:-1}"
SAFE_GPU_ID="${SAFE_GPU_ID:-TO_FILL_WHEN_SAFE}"
CONFIRM_ACCOUNT="${CONFIRM_BEDROCK_ACCOUNT_887730490125:-no}"
ALLOW_WHILE_H65_RUNNING="${ALLOW_WHILE_H65_RUNNING:-no}"
MIN_FREE_MIB="${MIN_FREE_MIB:-30000}"
LOG_DIR="/tmp/csd_h86_logs"
ARTIFACT_ROOT="outputs/generated/h86_smiles_qwen35_9b_isocyanates_train100_materialization_20260630"
PROVENANCE_DIR="${ARTIFACT_ROOT}/provenance_$(date -u +%Y%m%dT%H%M%SZ)"
PID_FILE="${LOG_DIR}/h86_smiles_qwen35_9b_isocyanates_train100_20260630.pid"
LOG_FILE="${LOG_DIR}/h86_smiles_qwen35_9b_isocyanates_train100_20260630.log"
MODEL="Qwen/Qwen3.5-9B"
TAG="qwen35_9b"
CLASS="isocyanates"
UTIL="0.55"
MINACC="0.92"
MINSYN="0.50"
NAME="smiles_${TAG}_${CLASS}_uv_qwen35_0627"
GEN_ROOT="outputs/generated/${NAME}"
OUT_JSON="outputs/controlled_comparison/smiles_${TAG}/${CLASS}/metadecode_uv.json"
PY="/apps/conda/aadivyar/envs/csd/bin/python"
H65_PID_FILE="/tmp/csd_h65_logs/h65_gsm9_timeoutguard_20260630.pid"

TRAIN_CMD=(
  "${PY}" -m synthesis.run_synthesis
  --task "Generate one new, valid, non-exemplar SMILES molecule for the ${CLASS} class. The answer contract is a single SMILES string and nothing else. Use the hidden parser-guided constrained chunk for that SMILES token sequence and avoid copying prompt exemplars."
  --dataset smiles
  --smiles-classes "${CLASS}"
  --smiles-samples-per-class 100
  --generation-model us.anthropic.claude-sonnet-4-6
  --generation-backend bedrock
  --anthropic-thinking enabled
  --anthropic-effort high
  --anthropic-thinking-display summarized
  --eval-model "${MODEL}"
  --eval-backend vllm
  --max-iterations 40
  --output-name "${NAME}"
  --output-dir "${GEN_ROOT}"
  --min-accuracy "${MINACC}"
  --min-syntax-rate "${MINSYN}"
  --eval-sample-size 100
  --eval-max-steps 400
  --eval-step-token-budget 1
  --eval-min-examples-before-threshold-stop 100
  --max-tokens 32768
  --restart-after-stuck-iters 0
  --vllm-gpu-memory-utilization "${UTIL}"
  --device auto
  --vllm-tensor-parallel-size 1
  --adaptive-helper-mask
  --helper-selection-policy bandit
  --refinement-beam-size 2
)

if [[ "${DRY_RUN}" == "1" ]]; then
  mkdir -p "${ARTIFACT_ROOT}"
  python - <<PY_H86_DRY_RUN
import json
from pathlib import Path
payload = {
  "hypothesis": "H86",
  "dry_run": True,
  "model_calls": 0,
  "gpu_calls": 0,
  "billed_api_calls": 0,
  "account_record": "887730490125",
  "scientific_target": "sample-size-aligned cold Qwen3.5-9B isocyanates SMILES UV matrix cell",
  "train_eval_sample_size": 100,
  "train_smiles_samples_per_class": 100,
  "heldout_sample_size": 100,
  "planned_output_json": "${OUT_JSON}",
  "planned_generated_root": "${GEN_ROOT}",
  "max_iterations": 40,
  "eval_model": "${MODEL}",
  "class": "${CLASS}",
  "util": float("${UTIL}"),
  "min_acc": float("${MINACC}"),
  "min_syn": float("${MINSYN}"),
  "uses_initial_strategy_file": False,
  "requires_real_launch_env": [
    "DRY_RUN=0",
    "SAFE_GPU_ID=<gpu>",
    "CONFIRM_BEDROCK_ACCOUNT_887730490125=yes"
  ],
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
Path("${ARTIFACT_ROOT}/h86_dry_run.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, sort_keys=True))
PY_H86_DRY_RUN
  printf 'H86 dry-run only; no paid/model/GPU/API call launched.\n'
  printf 'planned_log_file=%s\n' "${LOG_FILE}"
  printf 'planned_pid_file=%s\n' "${PID_FILE}"
  printf 'planned_provenance_dir=%s\n' "${PROVENANCE_DIR}"
  printf 'train_command='
  printf '%q ' "${TRAIN_CMD[@]}"
  printf '\n'
  exit 0
fi

if [[ "${CONFIRM_ACCOUNT}" != "yes" ]]; then
  printf 'ERROR: set CONFIRM_BEDROCK_ACCOUNT_887730490125=yes before real H86 launch.\n' >&2
  exit 2
fi
if [[ "${SAFE_GPU_ID}" == "TO_FILL_WHEN_SAFE" ]]; then
  printf 'ERROR: set SAFE_GPU_ID to an explicit GPU index before real H86 launch.\n' >&2
  exit 3
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

mkdir -p "${LOG_DIR}" "${PROVENANCE_DIR}/prelaunch" "${PROVENANCE_DIR}/postlaunch" "$(dirname "${OUT_JSON}")"
export CUDA_VISIBLE_DEVICES="${SAFE_GPU_ID}"
export LD_LIBRARY_PATH="/apps/conda/aadivyar/envs/csd/lib:${LD_LIBRARY_PATH:-}"
export CSD_CONSTRAINED_TEMPERATURE="0.7"
set -a
source /home/aadivyar/csd-generation/.env
set +a

if [[ -f "${OUT_JSON}" ]]; then
  cp "${OUT_JSON}" "${PROVENANCE_DIR}/prelaunch/metadecode_uv.prelaunch.json"
fi
if [[ -f "${GEN_ROOT}/latest_run.txt" ]]; then
  cp "${GEN_ROOT}/latest_run.txt" "${PROVENANCE_DIR}/prelaunch/latest_run.prelaunch.txt"
fi
if [[ -e "${GEN_ROOT}/latest" ]]; then
  readlink "${GEN_ROOT}/latest" > "${PROVENANCE_DIR}/prelaunch/latest_symlink_target.prelaunch.txt" 2>/dev/null || true
fi
python - <<PY_H86_PRELAUNCH_MANIFEST
import json
from pathlib import Path
payload = {
  "hypothesis": "H86",
  "prelaunch_snapshot_dir": "${PROVENANCE_DIR}/prelaunch",
  "postlaunch_snapshot_dir": "${PROVENANCE_DIR}/postlaunch",
  "generated_root": "${GEN_ROOT}",
  "heldout_json": "${OUT_JSON}",
  "train_eval_sample_size": 100,
  "heldout_sample_size": 100,
  "planned_train_command_note": "full command is emitted to the launch log and h86_dry_run.json",
}
Path("${PROVENANCE_DIR}/prelaunch/prelaunch_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY_H86_PRELAUNCH_MANIFEST

setsid bash -c '
set +e
provenance_dir="$1"
out_json="$2"
gen_root="$3"
py="$4"
model="$5"
class_name="$6"
util="$7"
shift 7
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
  csd="$("${py}" -c "import json,sys;print(json.load(open(sys.argv[1])).get(\"compiled_dir\", \"\"))" "${latest_run}/results/success_report.json" 2>/dev/null)"
  if [[ -n "${csd}" && -s "${csd}/GeneratedCSD.py" ]]; then
    "${py}" -m synthesis.scripts.reevaluate_compiled_csd "${csd}/GeneratedCSD.py" \
      --dataset smiles --smiles-classes "${class_name}" \
      --eval-model "${model}" --eval-backend vllm \
      --sample-size 100 --max-steps 400 --step-token-budget 1 \
      --vllm-gpu-memory-utilization "${util}" \
      --output-json "${out_json}"
    status=$?
    if [[ -f "${out_json}" ]]; then
      cp "${out_json}" "${provenance_dir}/postlaunch/metadecode_uv.postlaunch.json"
    fi
  fi
fi
if [[ -e "${gen_root}/latest" ]]; then
  readlink "${gen_root}/latest" > "${provenance_dir}/postlaunch/latest_symlink_target.postlaunch.txt" 2>/dev/null || true
fi
printf "%s\n" "${status}" > "${provenance_dir}/postlaunch/exit_status.txt"
exit "${status}"
' h86_provenance_wrapper "${PROVENANCE_DIR}" "${OUT_JSON}" "${GEN_ROOT}" "${PY}" "${MODEL}" "${CLASS}" "${UTIL}" "${TRAIN_CMD[@]}" >"${LOG_FILE}" 2>&1 < /dev/null &
echo "$!" > "${PID_FILE}"
printf 'Launched H86 pid=%s log=%s pid_file=%s provenance_dir=%s\n' "$(cat "${PID_FILE}")" "${LOG_FILE}" "${PID_FILE}" "${PROVENANCE_DIR}"
