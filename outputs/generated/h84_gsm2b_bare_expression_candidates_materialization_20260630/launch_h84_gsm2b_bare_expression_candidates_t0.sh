#!/usr/bin/env bash
set -euo pipefail

# H84 is a local no-billing GSM-2B bare-expression candidate smoke.
# It dry-runs by default.
#
# To launch for real after a clean GPU is available:
#   DRY_RUN=0 SAFE_GPU_ID=<gpu_index> ./launch_h84_gsm2b_bare_expression_candidates_t0.sh

REPO_ROOT="${REPO_ROOT:-/home/aadivyar/csd-generation}"
WORKTREE_ROOT="${WORKTREE_ROOT:-/home/aadivyar/.config/superpowers/worktrees/csd-generation/h31-candidate-consensus}"
cd "${WORKTREE_ROOT}"

DRY_RUN="${DRY_RUN:-1}"
SAFE_GPU_ID="${SAFE_GPU_ID:-TO_FILL_WHEN_SAFE}"
MIN_FREE_MIB="${MIN_FREE_MIB:-12000}"

LOG_DIR="/tmp/csd_h84_logs"
PID_FILE="${LOG_DIR}/h84_gsm2b_bare_expression_candidates_t0_20260630.pid"
LOG_FILE="${LOG_DIR}/h84_gsm2b_bare_expression_candidates_t0_20260630.log"
WORKER_FILE="${LOG_DIR}/h84_gsm2b_bare_expression_candidates_t0_worker.sh"

OUTPUT_NAME="h84_gsm2b_bare_expression_candidates_20260630_t0"
SOURCE_ID="h84_gsm2b_bare_expression_candidates_t0"
STRATEGY_FILE="${REPO_ROOT}/saved-results/2026-06-30-h84-gsm2b-bare-expression-candidates-body.dfy"
SPLIT_FILE="${REPO_ROOT}/environment/benchmark_splits/gsm_symbolic_crane_proportional_49x49_seed123.json"
OUT_ROOT="${REPO_ROOT}/outputs/generated/${OUTPUT_NAME}"
SUCCESS_JSON="${OUT_ROOT}/results/direct_eval_success.json"
FAILURE_JSON="${OUT_ROOT}/results/direct_eval_failure.json"
STRUCTURED_JSON="${OUT_ROOT}/results/structured_bare_expression_candidates.json"
MATERIALIZATION_DIR="${REPO_ROOT}/outputs/generated/h84_gsm2b_bare_expression_candidates_materialization_20260630"
DRY_RUN_JSON="${MATERIALIZATION_DIR}/direct_eval_dry_run.json"
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
  mkdir -p "${MATERIALIZATION_DIR}"
  "${CMD[@]}" --dry-run --dry-run-output "${DRY_RUN_JSON}"
  printf 'H84 dry run only; no GPU/model/API call launched.\n'
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
  printf 'ERROR: set SAFE_GPU_ID to an explicit GPU index before launching H84.\n' >&2
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
# The direct-eval runner strips paid-provider environment variables before imports.

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
import re
from pathlib import Path

report_path = Path(os.environ["SUCCESS_JSON"])
out_path = Path(os.environ["STRUCTURED_JSON"])
report = json.loads(report_path.read_text())

span_re = re.compile(r"<<\s*(.*?)\s*>>", re.DOTALL)
allowed_re = re.compile(r"^[A-Za-z0-9_+\-*/%(),. \t]+$")
bad_word_re = re.compile(
    r"\b(candidate|route|because|therefore|answer|total|remaining|dollars|people|cars|miles|hours|minutes|liters|apples|boxes|students)\b",
    re.I,
)

def clean_expr(value):
    return " ".join(str(value).split())

def parseable_ish(expr):
    expr = clean_expr(expr)
    if not expr or len(expr) > 240:
        return False
    if "<<" in expr or ">>" in expr or "=" in expr:
        return False
    if not allowed_re.match(expr):
        return False
    if bad_word_re.search(expr):
        return False
    return bool(re.search(r"[A-Za-z0-9]", expr))

candidates = []
seen = set()
for sample_index, sample in enumerate(report.get("sample_outputs") or [], start=1):
    for field_name in ("full_output", "scored_output"):
        text = sample.get(field_name) or ""
        for span_index, span in enumerate(span_re.findall(text), start=1):
            expr = clean_expr(span)
            key = (sample_index, expr)
            if not expr or key in seen:
                continue
            seen.add(key)
            is_parseable = parseable_ish(expr)
            candidates.append({
                "candidate_kind": "visible_output_span",
                "candidate_source": f"{os.environ['SOURCE_ID']}:{field_name}:{span_index}",
                "expression": expr,
                "equivalence_key": expr,
                "group_id": sample_index,
                "sample_index": sample_index,
                "source_id": os.environ["SOURCE_ID"],
                "source_family": "h84_gsm2b_bare_expression_candidates",
                "output_name": os.environ["OUTPUT_NAME"],
                "quality_score": 0.8 if is_parseable else 0.2,
                "scorer_metadata": {
                    "field_name": field_name,
                    "parseable_ish": is_parseable,
                    "uses_expected_or_correctness": False,
                },
            })

artifact = {
    "source_id": os.environ["SOURCE_ID"],
    "output_name": os.environ["OUTPUT_NAME"],
    "selection_uses_gold": False,
    "candidate_count": len(candidates),
    "parseable_ish_candidate_count": sum(1 for c in candidates if c["scorer_metadata"]["parseable_ish"]),
    "group_count": len({c["group_id"] for c in candidates}),
    "parseable_ish_group_count": len({c["group_id"] for c in candidates if c["scorer_metadata"]["parseable_ish"]}),
    "candidates": candidates,
    "postprocess": {
        "source_report": str(report_path),
        "candidate_source": "visible_spans_from_full_output_and_scored_output",
        "uses_expected_or_correctness": False,
    },
}
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
print(f"WROTE_STRUCTURED_BARE_EXPRESSION_CANDIDATES={out_path}")
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
printf 'Launched H84 pid=%s log=%s pid_file=%s\n' "$(cat "${PID_FILE}")" "${LOG_FILE}" "${PID_FILE}"
