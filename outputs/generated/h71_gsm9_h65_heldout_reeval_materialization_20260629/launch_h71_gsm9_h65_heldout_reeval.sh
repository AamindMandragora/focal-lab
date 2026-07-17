#!/usr/bin/env bash
set -euo pipefail

# H71: future pure held-out re-eval for the H65 GSM-9B strategy.
# Dry-run by default. Real launch is allowed only after H65 writes a train
# success report and its synthesis PID is no longer alive.

REPO_ROOT="${REPO_ROOT:-/home/aadivyar/csd-generation}"
cd "${REPO_ROOT}"
DRY_RUN="${DRY_RUN:-1}"
SAFE_GPU_ID="${SAFE_GPU_ID:-TO_FILL_WHEN_SAFE}"
MIN_FREE_MIB="${MIN_FREE_MIB:-24000}"
H65_ROOT="outputs/generated/synth_gsm_9b_z3fix_seed123train_h65_timeoutguard_20260630"
H65_PID_FILE="/tmp/csd_h65_logs/h65_gsm9_timeoutguard_20260630.pid"
H71_ROOT="outputs/generated/h71_gsm9_h65_heldout_reeval_20260629"
LOG_DIR="/tmp/csd_h71_logs"
PID_FILE="${LOG_DIR}/h71_gsm9_h65_heldout_reeval_20260629.pid"
LOG_FILE="${LOG_DIR}/h71_gsm9_h65_heldout_reeval_20260629.log"
SPLIT_FILE="/home/aadivyar/csd-generation/environment/benchmark_splits/gsm_symbolic_crane_proportional_49x49_seed123.json"
SUCCESS_REPORT=""
STRATEGY_BODY=""

if [[ -f "${H65_ROOT}/latest_run.txt" ]]; then
  LATEST_RUN="$(cat "${H65_ROOT}/latest_run.txt")"
  if [[ -f "${LATEST_RUN}/results/success_report.json" ]]; then
    SUCCESS_REPORT="${LATEST_RUN}/results/success_report.json"
  fi
fi

if [[ -n "${SUCCESS_REPORT}" ]]; then
  STRATEGY_BODY="$(python - "${SUCCESS_REPORT}" <<'PY_RESOLVE_STRATEGY'
import json, sys
from pathlib import Path
report = json.loads(Path(sys.argv[1]).read_text())
for key in ("dafny_file_canonical", "dafny_file"):
    value = report.get(key)
    if value and Path(value).exists():
        print(value)
        break
else:
    print("")
PY_RESOLVE_STRATEGY
)"
fi

CMD=(
  python -m synthesis.run_synthesis
  --task "Solve math word problems step by step, wrapping intermediate symbolic expressions and the final answer inside << >> delimiters."
  --dataset gsm_symbolic
  --max-iterations 1
  --initial-strategy-file "${STRATEGY_BODY:-H65_SUCCESS_STRATEGY_TO_BE_RESOLVED}"
  --generation-model us.anthropic.claude-sonnet-4-6
  --generation-backend bedrock
  --eval-model Qwen/Qwen3.5-9B
  --eval-backend vllm
  --output-name h71_gsm9_h65_heldout_reeval_20260629
  --min-accuracy 0.0
  --min-syntax-rate 0.0
  --eval-sample-size 49
  --eval-max-steps 900
  --eval-step-token-budget 1
  --eval-max-seconds-per-example 600
  --eval-min-examples-before-threshold-stop 49
  --vllm-gpu-memory-utilization 0.55
  --device auto
  --output-dir "${H71_ROOT}"
  --adaptive-helper-mask
  --helper-selection-policy bandit
  --anthropic-thinking enabled
  --anthropic-effort high
  --anthropic-thinking-display summarized
  --vllm-tensor-parallel-size 1
  --gsm-split-file "${SPLIT_FILE}"
  --gsm-split-name eval
)

if [[ "${DRY_RUN}" == "1" ]]; then
  python - <<PY_DRY_RUN_JSON
import json
from pathlib import Path
payload = {
  "hypothesis": "H71",
  "dry_run": True,
  "model_calls": 0,
  "gpu_calls": 0,
  "billed_api_calls": 0,
  "h65_root": "${H65_ROOT}",
  "h65_success_report_found": bool("${SUCCESS_REPORT}"),
  "resolved_strategy_body": "${STRATEGY_BODY}",
  "planned_output_root": "${H71_ROOT}",
  "planned_pid_file": "${PID_FILE}",
  "planned_log_file": "${LOG_FILE}",
  "max_iterations": 1,
  "eval_model": "Qwen/Qwen3.5-9B",
  "gsm_split_name": "eval",
  "gsm_split_file": "${SPLIT_FILE}",
  "eval_max_seconds_per_example": 600,
  "min_accuracy": 0.0,
  "min_syntax_rate": 0.0,
  "requires_real_launch_env": ["DRY_RUN=0", "SAFE_GPU_ID=<gpu>"],
  "real_launch_guards": ["H65 success_report exists", "H65 PID not alive", "explicit GPU", "free memory >= MIN_FREE_MIB", "no non-aadivyar process on chosen GPU"],
}
Path("outputs/generated/h71_gsm9_h65_heldout_reeval_materialization_20260629/h71_dry_run.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, sort_keys=True))
PY_DRY_RUN_JSON
  printf 'H71 dry-run only; no paid/model/GPU/API call launched.\n'
  printf 'planned_command='
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

if [[ -z "${SUCCESS_REPORT}" ]]; then
  printf 'ERROR: H65 train success_report.json is not present; refusing H71 held-out re-eval.\n' >&2
  exit 2
fi
if [[ -z "${STRATEGY_BODY}" || ! -f "${STRATEGY_BODY}" ]]; then
  printf 'ERROR: could not resolve an existing H65 strategy body from %s.\n' "${SUCCESS_REPORT}" >&2
  exit 3
fi
if [[ -f "${H65_PID_FILE}" ]]; then
  H65_PID="$(cat "${H65_PID_FILE}")"
  if ps -p "${H65_PID}" >/dev/null 2>&1; then
    printf 'ERROR: H65 pid=%s is still running; refusing held-out re-eval until synthesis finishes.\n' "${H65_PID}" >&2
    exit 4
  fi
fi
if [[ "${SAFE_GPU_ID}" == "TO_FILL_WHEN_SAFE" ]]; then
  printf 'ERROR: set SAFE_GPU_ID to an explicit GPU index before real H71 launch.\n' >&2
  exit 5
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
NON_AADIVYAR="$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits | while IFS=',' read -r uuid pid; do
  idx=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits | awk -F', *' -v u="${uuid// /}" '$2 == u {print $1}')
  if [[ "${idx}" == "${SAFE_GPU_ID}" ]]; then
    user=$(ps -o user= -p "${pid// /}" | tr -d ' ')
    if [[ -n "${user}" && "${user}" != "aadivyar" ]]; then
      printf '%s:%s ' "${pid// /}" "${user}"
    fi
  fi
done)"
if [[ -n "${NON_AADIVYAR}" ]]; then
  printf 'ERROR: GPU %s has non-aadivyar process(es): %s\n' "${SAFE_GPU_ID}" "${NON_AADIVYAR}" >&2
  exit 8
fi
mkdir -p "${LOG_DIR}" "${H71_ROOT}"
CUDA_VISIBLE_DEVICES="${SAFE_GPU_ID}" nohup "${CMD[@]}" >"${LOG_FILE}" 2>&1 &
printf '%s\n' "$!" >"${PID_FILE}"
printf 'launched_h71_pid=%s\n' "$!"
printf 'log=%s\n' "${LOG_FILE}"
printf 'pid_file=%s\n' "${PID_FILE}"
