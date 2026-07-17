#!/usr/bin/env bash
set -euo pipefail

# H64 is a local no-billing GSM-2B structured-candidate smoke.
# It dry-runs by default and must not run before H52 unless a future operator
# explicitly chooses a safe GPU for this lower-priority smoke.
#
# To launch for real:
#   DRY_RUN=0
#   SAFE_GPU_ID=<gpu_index>

REPO_ROOT="${REPO_ROOT:-/home/aadivyar/csd-generation}"
WORKTREE_ROOT="${WORKTREE_ROOT:-/home/aadivyar/.config/superpowers/worktrees/csd-generation/h31-candidate-consensus}"
cd "${WORKTREE_ROOT}"

DRY_RUN="${DRY_RUN:-1}"
SAFE_GPU_ID="${SAFE_GPU_ID:-TO_FILL_WHEN_SAFE}"
MIN_FREE_MIB="${MIN_FREE_MIB:-12000}"

LOG_DIR="/tmp/csd_h64_logs"
PID_FILE="${LOG_DIR}/h64_gsm2b_named_route_t0_20260629.pid"
LOG_FILE="${LOG_DIR}/h64_gsm2b_named_route_t0_20260629.log"
WORKER_FILE="${LOG_DIR}/h64_gsm2b_named_route_t0_worker.sh"

OUTPUT_NAME="h64_gsm2b_named_route_structured_candidates_20260629_t0"
SOURCE_ID="h64_gsm2b_named_route_t0"
STRATEGY_FILE="${REPO_ROOT}/saved-results/2026-06-29-h64-gsm2b-named-route-candidates-body.dfy"
SPLIT_FILE="${REPO_ROOT}/environment/benchmark_splits/gsm_symbolic_crane_proportional_49x49_seed123.json"
OUT_ROOT="${REPO_ROOT}/outputs/generated/${OUTPUT_NAME}"
SUCCESS_JSON="${OUT_ROOT}/results/direct_eval_success.json"
FAILURE_JSON="${OUT_ROOT}/results/direct_eval_failure.json"
STRUCTURED_JSON="${OUT_ROOT}/results/structured_candidates.json"
DRY_RUN_JSON="${REPO_ROOT}/outputs/generated/h64_gsm2b_named_route_structured_candidates_materialization_20260629/direct_eval_dry_run.json"
PY="/apps/conda/aadivyar/envs/csd/bin/python"

CMD=(
  "${PY}" -m synthesis.evaluate.parameterized_direct_eval
  --repo-root "${REPO_ROOT}"
  --strategy-file "${STRATEGY_FILE}"
  --output-name "${OUTPUT_NAME}"
  --source-id "${SOURCE_ID}"
  --eval-model Qwen/Qwen3.5-2B
  --dataset gsm_symbolic
  --gsm-split-file "${SPLIT_FILE}"
  --gsm-split-name train
  --sample-size 49
  --max-steps 900
  --max-seconds-per-example 600
  --vllm-gpu-memory-utilization 0.20
)

if [[ "${DRY_RUN}" == "1" ]]; then
  "${CMD[@]}" --dry-run --dry-run-output "${DRY_RUN_JSON}"
  printf 'H64 dry run only; no GPU/model/API call launched.\n'
  printf 'planned_output_root=%s\n' "${OUT_ROOT}"
  printf 'planned_success_json=%s\n' "${SUCCESS_JSON}"
  printf 'planned_failure_json=%s\n' "${FAILURE_JSON}"
  printf 'planned_structured_json=%s\n' "${STRUCTURED_JSON}"
  printf 'planned_pid_file=%s\n' "${PID_FILE}"
  printf 'planned_log_file=%s\n' "${LOG_FILE}"
  printf 'planned_command='
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

if [[ "${SAFE_GPU_ID}" == "TO_FILL_WHEN_SAFE" ]]; then
  printf 'ERROR: set SAFE_GPU_ID to an explicit GPU index before launching H64.\n' >&2
  exit 2
fi

GPU_FREE_MIB="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', *' -v gpu="${SAFE_GPU_ID}" '$1 == gpu {print $2}')"
if [[ -z "${GPU_FREE_MIB}" ]]; then
  printf 'ERROR: GPU index %s not found.\n' "${SAFE_GPU_ID}" >&2
  exit 3
fi
if (( GPU_FREE_MIB < MIN_FREE_MIB )); then
  printf 'ERROR: GPU %s has only %s MiB free; need at least %s MiB.\n' "${SAFE_GPU_ID}" "${GPU_FREE_MIB}" "${MIN_FREE_MIB}" >&2
  exit 4
fi

GPU_UUID="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits | awk -F', *' -v gpu="${SAFE_GPU_ID}" '$1 == gpu {print $2}')"
while IFS=',' read -r app_gpu_uuid app_pid _app_name _used_mem; do
  app_gpu_uuid="$(printf '%s' "${app_gpu_uuid}" | xargs)"
  app_pid="$(printf '%s' "${app_pid}" | xargs)"
  [[ -z "${app_gpu_uuid}" || "${app_gpu_uuid}" != "${GPU_UUID}" ]] && continue
  app_user="$(ps -o user= -p "${app_pid}" | xargs || true)"
  if [[ -n "${app_user}" && "${app_user}" != "aadivyar" ]]; then
    printf 'ERROR: GPU %s has non-aadivyar process pid=%s user=%s.\n' "${SAFE_GPU_ID}" "${app_pid}" "${app_user}" >&2
    exit 5
  fi
done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits || true)

mkdir -p "${LOG_DIR}"
cat > "${WORKER_FILE}" <<'WORKER'
#!/usr/bin/env bash
set -euo pipefail

cd "${WORKTREE_ROOT}"
export CUDA_VISIBLE_DEVICES="${SAFE_GPU_ID}"
export LD_LIBRARY_PATH="/apps/conda/aadivyar/envs/csd/lib:${LD_LIBRARY_PATH:-}"
# The Python runner strips paid-provider environment variables before imports.

"${PY}" -m synthesis.evaluate.parameterized_direct_eval \
  --repo-root "${REPO_ROOT}" \
  --strategy-file "${STRATEGY_FILE}" \
  --output-name "${OUTPUT_NAME}" \
  --source-id "${SOURCE_ID}" \
  --eval-model Qwen/Qwen3.5-2B \
  --dataset gsm_symbolic \
  --gsm-split-file "${SPLIT_FILE}" \
  --gsm-split-name train \
  --sample-size 49 \
  --max-steps 900 \
  --max-seconds-per-example 600 \
  --vllm-gpu-memory-utilization 0.20

if [[ -f "${SUCCESS_JSON}" ]]; then
  "${PY}" - <<'PYCODE'
import json
import os
from pathlib import Path

from synthesis.evaluate.candidate_line_scorer import load_gsm_split_examples
from synthesis.evaluate.candidate_report_adapter import structured_candidate_artifact_from_direct_eval_report

report_path = Path(os.environ["SUCCESS_JSON"])
split_file = Path(os.environ["SPLIT_FILE"])
out_path = Path(os.environ["STRUCTURED_JSON"])

report = json.loads(report_path.read_text())
examples = load_gsm_split_examples(split_file, split_name="train")
artifact = structured_candidate_artifact_from_direct_eval_report(
    report,
    examples,
    include_candidate_lines=True,
    source_family="h64_gsm2b_named_route",
)
artifact["postprocess"] = {
    "selection_uses_gold": False,
    "source_report": str(report_path),
    "split_file": str(split_file),
    "split_name": "train",
    "include_candidate_lines": True,
}
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
print(f"WROTE_STRUCTURED_CANDIDATES={out_path}")
PYCODE
fi
WORKER
chmod +x "${WORKER_FILE}"

setsid env \
  REPO_ROOT="${REPO_ROOT}" \
  WORKTREE_ROOT="${WORKTREE_ROOT}" \
  SAFE_GPU_ID="${SAFE_GPU_ID}" \
  PY="${PY}" \
  STRATEGY_FILE="${STRATEGY_FILE}" \
  OUTPUT_NAME="${OUTPUT_NAME}" \
  SOURCE_ID="${SOURCE_ID}" \
  SPLIT_FILE="${SPLIT_FILE}" \
  SUCCESS_JSON="${SUCCESS_JSON}" \
  STRUCTURED_JSON="${STRUCTURED_JSON}" \
  "${WORKER_FILE}" >"${LOG_FILE}" 2>&1 < /dev/null &
echo "$!" > "${PID_FILE}"
printf 'Launched H64 pid=%s log=%s pid_file=%s\n' "$(cat "${PID_FILE}")" "${LOG_FILE}" "${PID_FILE}"
