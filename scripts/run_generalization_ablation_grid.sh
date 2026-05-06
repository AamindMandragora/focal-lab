#!/usr/bin/env bash
set -euo pipefail

cd /home/aadivyar/csd-generation
export PYTHONPATH="/home/aadivyar/csd-generation:${PYTHONPATH:-}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-generalization_ablation_grid_${RUN_STAMP}}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/aadivyar/csd-generation/outputs/generated-csd}"
LOG_DIR="${OUTPUT_DIR}/logs/${RUN_NAME}"
SUMMARY_PATH="${OUTPUT_DIR}/benchmarks/${RUN_NAME}_summary.jsonl"
mkdir -p "${LOG_DIR}" "$(dirname "${SUMMARY_PATH}")"

read -r -a DATASET_VALUES <<< "${DATASET_VALUES:-gsm spider smiles}"
read -r -a MAX_STEPS_VALUES <<< "${MAX_STEPS_VALUES:-256 512 1024}"
read -r -a SYNTHESIS_ITERATIONS_VALUES <<< "${SYNTHESIS_ITERATIONS_VALUES:-5 10 15 20}"
ABLATION_SWEEP="${ABLATION_SWEEP:-both}"
FIXED_MAX_STEPS="${FIXED_MAX_STEPS:-512}"
FIXED_SYNTHESIS_ITERATIONS="${FIXED_SYNTHESIS_ITERATIONS:-20}"

GENERATION_MODEL="${GENERATION_MODEL:-gpt-5.4}"
GENERATION_BACKEND="${GENERATION_BACKEND:-openai}"
EVAL_MODEL="${EVAL_MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}"
EVAL_BACKEND="${EVAL_BACKEND:-vllm}"
DEVICE="${DEVICE:-cuda}"
CRANE_REPO="${CRANE_REPO:-/home/aadivyar/CRANE}"
CRANE_DEVICE="${CRANE_DEVICE:-cuda:0}"
KILL_VLLM_WORKERS="${KILL_VLLM_WORKERS:-1}"
ORIGINAL_FRAMEWORK_CUDA_VISIBLE_DEVICES="${ORIGINAL_FRAMEWORK_CUDA_VISIBLE_DEVICES:-auto}"
GPU_MIN_FREE_MIB="${GPU_MIN_FREE_MIB:-12000}"
GPU_AVOID_DEVICES="${GPU_AVOID_DEVICES:-}"
CARS_CUDA_VISIBLE_DEVICES="${CARS_CUDA_VISIBLE_DEVICES:-auto}"

GSM_SPLIT_FILE="${GSM_SPLIT_FILE:-${OUTPUT_DIR}/splits/gsm_absolute_rubric_seed123_train50_eval50.json}"
GSM_SOURCE_DIR="${GSM_SOURCE_DIR:-/home/aadivyar/CRANE/src/gsm_symbolic}"
SPIDER_SPLIT_FILE="${SPIDER_SPLIT_FILE:-${OUTPUT_DIR}/splits/spider_seed123_train50_test100.json}"
SPIDER_SOURCE="${SPIDER_SOURCE:-local}"
SMILES_CLASSES="${SMILES_CLASSES:-acrylates,chain_extenders,isocyanates}"

echo "[ablation-grid] run_name=${RUN_NAME}"
echo "[ablation-grid] datasets=${DATASET_VALUES[*]}"
echo "[ablation-grid] sweep=${ABLATION_SWEEP}"
echo "[ablation-grid] max_steps_sweep=${MAX_STEPS_VALUES[*]} fixed_iterations=${FIXED_SYNTHESIS_ITERATIONS}"
echo "[ablation-grid] iterations_sweep=${SYNTHESIS_ITERATIONS_VALUES[*]} fixed_max_steps=${FIXED_MAX_STEPS}"
echo "[ablation-grid] summary=${SUMMARY_PATH}"

kill_vllm_workers() {
  local reason="$1"
  local -a pids
  mapfile -t pids < <(
    { pgrep -u "$(id -u)" -af 'vllm|VLLM|multiproc_worker_utils|VllmWorkerProcess' || true; } \
      | awk '/pgrep/ {next} /run_generalization_ablation_grid/ {next} {print $1}'
  )
  if [[ "${#pids[@]}" -eq 0 ]]; then
    return 0
  fi
  echo "[vllm-cleanup] ${reason}: terminating owned vLLM worker pids=${pids[*]}"
  kill "${pids[@]}" 2>/dev/null || true
  sleep 2
  kill -9 "${pids[@]}" 2>/dev/null || true
}

if [[ "${KILL_VLLM_WORKERS}" == "1" ]]; then
  kill_vllm_workers "before ablation grid"
else
  echo "[vllm-cleanup] disabled for this ablation grid"
fi

run_cell() {
  local dataset="$1"
  local max_steps="$2"
  local iterations="$3"
  local sweep="$4"
  local cell_id="${dataset}_ablation_${sweep}_steps${max_steps}_iters${iterations}"
  local cell_run_name="${RUN_NAME}_${cell_id}"
  local log_path="${LOG_DIR}/${cell_id}.log"
  local started_at
  started_at="$(date --iso-8601=seconds)"
  local -a cmd

  case "${dataset}" in
    gsm)
      cmd=(/opt/anaconda/bin/python scripts/gsm_split_synthesis_workflow.py run-all
        --run-name "${cell_run_name}"
        --split-file "${GSM_SPLIT_FILE}"
        --split-strategy stratified
        --difficulty-train-counts easy=13,medium=12,hard=25
        --difficulty-eval-counts easy=13,medium=12,hard=25
        --gsm-source-dir "${GSM_SOURCE_DIR}"
        --output-dir "${OUTPUT_DIR}"
        --eval-model "${EVAL_MODEL}"
        --eval-backend "${EVAL_BACKEND}"
        --device "${DEVICE}"
        --max-iterations "${iterations}"
        --generation-model "${GENERATION_MODEL}"
        --generation-backend "${GENERATION_BACKEND}"
        --eval-max-steps "${max_steps}"
        --eval-step-token-budget 1
        --vllm-gpu-memory-utilization 0.5
        --vllm-max-model-len 8192
        --synthesis-max-tokens 6144
        --itergen-repo /home/aadivyar/itergen
        --itergen-device cuda:0
        --original-framework-cuda-visible-devices "${ORIGINAL_FRAMEWORK_CUDA_VISIBLE_DEVICES}"
        --gpu-min-free-mib "${GPU_MIN_FREE_MIB}"
        --gpu-avoid-devices "${GPU_AVOID_DEVICES}"
        --crane-repo "${CRANE_REPO}"
        --crane-device "${CRANE_DEVICE}"
        --cars-repo /home/aadivyar/cars
        --cars-style cars
        --cars-max-attempts-per-example 2000
        --cars-cuda-visible-devices "${CARS_CUDA_VISIBLE_DEVICES}")
      ;;
    spider|sql)
      dataset="spider"
      cell_id="spider_ablation_${sweep}_steps${max_steps}_iters${iterations}"
      cell_run_name="${RUN_NAME}_${cell_id}"
      log_path="${LOG_DIR}/${cell_id}.log"
      cmd=(/opt/anaconda/bin/python scripts/itergen_generalization_workflow.py
        --run-name "${cell_run_name}"
        --output-dir "${OUTPUT_DIR}"
        --itergen-repo /home/aadivyar/itergen
        --split-file "${SPIDER_SPLIT_FILE}"
        --spider-source "${SPIDER_SOURCE}"
        --train-size 50
        --test-size 100
        --itergen-model "${EVAL_MODEL}"
        --eval-model "${EVAL_MODEL}"
        --eval-backend "${EVAL_BACKEND}"
        --device "${DEVICE}"
	        --itergen-device cuda:0
	        --original-framework-cuda-visible-devices "${ORIGINAL_FRAMEWORK_CUDA_VISIBLE_DEVICES}"
	        --gpu-min-free-mib "${GPU_MIN_FREE_MIB}"
	        --gpu-avoid-devices "${GPU_AVOID_DEVICES}"
	        --crane-repo "${CRANE_REPO}"
	        --crane-device "${CRANE_DEVICE}"
	        --max-iterations "${iterations}"
        --generation-model "${GENERATION_MODEL}"
        --generation-backend "${GENERATION_BACKEND}"
        --eval-max-steps "${max_steps}"
        --eval-step-token-budget 4
        --vllm-gpu-memory-utilization 0.75
        --vllm-max-model-len 4096
        --synthesis-max-tokens 6144
        --cars-repo /home/aadivyar/cars
        --cars-style cars
        --cars-max-attempts-per-example 2000
        --cars-cuda-visible-devices "${CARS_CUDA_VISIBLE_DEVICES}")
      ;;
    smiles)
      cmd=(/opt/anaconda/bin/python scripts/smiles_generalization_workflow.py
        --run-name "${cell_run_name}"
        --output-dir "${OUTPUT_DIR}"
        --cars-repo /home/aadivyar/cars
        --classes "${SMILES_CLASSES}"
        --train-samples 50
        --test-samples 100
        --eval-model "${EVAL_MODEL}"
        --eval-backend "${EVAL_BACKEND}"
        --device "${DEVICE}"
	        --cuda-visible-devices "${CARS_CUDA_VISIBLE_DEVICES}"
	        --crane-repo "${CRANE_REPO}"
	        --crane-device "${CRANE_DEVICE}"
	        --original-framework-cuda-visible-devices "${ORIGINAL_FRAMEWORK_CUDA_VISIBLE_DEVICES}"
	        --gpu-min-free-mib "${GPU_MIN_FREE_MIB}"
	        --gpu-avoid-devices "${GPU_AVOID_DEVICES}"
	        --model-number 2
        --cars-style cars
        --max-attempts 2000
        --max-iterations "${iterations}"
        --generation-model "${GENERATION_MODEL}"
        --generation-backend "${GENERATION_BACKEND}"
        --eval-max-steps "${max_steps}"
        --eval-step-token-budget 1
        --vllm-gpu-memory-utilization 0.75
        --vllm-max-model-len 4096
        --synthesis-max-tokens 6144
        --itergen-repo /home/aadivyar/itergen
        --itergen-device cuda:0)
      ;;
    *)
      echo "[ablation-grid:error] unknown dataset=${dataset}" >&2
      return 2
      ;;
  esac

  echo "[ablation-grid:start] ${cell_id} ${started_at}"
  printf '[ablation-grid:cmd] '
  printf '%q ' "${cmd[@]}"
  printf '\n'

  set +e
  if [[ "${KILL_VLLM_WORKERS}" == "1" ]]; then
    kill_vllm_workers "before ${cell_id}"
  fi
  "${cmd[@]}" >"${log_path}" 2>&1
  local rc=$?
  set -e

  local finished_at status
  finished_at="$(date --iso-8601=seconds)"
  status="completed"
  if [[ "${rc}" -ne 0 ]]; then
    status="failed"
  fi
/opt/anaconda/bin/python - <<PY
import json
from pathlib import Path

dataset = "${dataset}"
output_dir = Path("${OUTPUT_DIR}")
cell_run_name = "${cell_run_name}"

def load_json(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None

def metric_value(payload, *keys):
    cur = payload
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur

def add_gsm_metrics(row, summary):
    comparison = summary.get("heldout_comparison") or {}
    results = comparison.get("results") or []
    baseline = results[0] if len(results) > 0 and isinstance(results[0], dict) else {}
    csd = results[1] if len(results) > 1 and isinstance(results[1], dict) else {}
    row.update({
        "metadecode_accuracy": csd.get("accuracy"),
        "metadecode_syntax_rate": csd.get("syntax_rate"),
        "metadecode_accuracy_denominator": csd.get("accuracy_denominator"),
        "baseline_crane_accuracy": baseline.get("accuracy"),
        "baseline_crane_syntax_rate": baseline.get("syntax_rate"),
        "heldout_delta_accuracy": metric_value(comparison, "delta", "accuracy"),
        "heldout_delta_syntax_rate": metric_value(comparison, "delta", "syntax_rate"),
        "accuracy": csd.get("accuracy"),
        "syntax_rate": csd.get("syntax_rate"),
    })

def add_spider_metrics(row, summary):
    results = summary.get("results") or {}
    row.update(results)
    row.update({
        "metadecode_accuracy": results.get("csd_test_accuracy"),
        "baseline_itergen_accuracy": results.get("itergen_test_accuracy"),
        "min_accuracy": summary.get("min_accuracy"),
        "min_syntax_rate": summary.get("min_syntax_rate"),
        "accuracy": results.get("csd_test_accuracy"),
    })

def normalize_smiles_metric(path):
    payload = load_json(path)
    if not isinstance(payload, dict):
        return {}
    rows = payload.get("csd") or payload.get("classes") or []
    if isinstance(rows, list) and rows:
        return dict(rows[0])
    return dict(payload)

def add_smiles_metrics(row, summary):
    classes = summary.get("classes") or []
    total_correct = 0
    total_denominator = 0
    total_attempts = 0
    syntax_pass = 0.0
    class_metrics = []
    for class_summary in classes:
        if not isinstance(class_summary, dict):
            continue
        metric = normalize_smiles_metric(class_summary.get("csd_benchmark"))
        if not metric:
            continue
        denom = int(metric.get("accuracy_denominator") or 0)
        correct = int(metric.get("accuracy_num_correct") or 0)
        attempts = int(metric.get("attempt_count") or len(metric.get("records") or []) or 0)
        syntax_rate = float(metric.get("syntax_rate") or 0.0)
        total_correct += correct
        total_denominator += denom
        total_attempts += attempts
        syntax_pass += syntax_rate * attempts
        class_metrics.append({
            "class_name": class_summary.get("class_name") or metric.get("class_name"),
            "accuracy": metric.get("accuracy"),
            "syntax_rate": metric.get("syntax_rate"),
            "accuracy_num_correct": metric.get("accuracy_num_correct"),
            "accuracy_denominator": metric.get("accuracy_denominator"),
            "invalid_outputs_excluded_from_accuracy": metric.get("invalid_outputs_excluded_from_accuracy"),
        })
    row.update({
        "smiles_class_metrics": class_metrics,
        "metadecode_accuracy": (total_correct / total_denominator) if total_denominator else None,
        "metadecode_syntax_rate": (syntax_pass / total_attempts) if total_attempts else None,
        "metadecode_accuracy_num_correct": total_correct,
        "metadecode_accuracy_denominator": total_denominator,
        "accuracy": (total_correct / total_denominator) if total_denominator else None,
        "syntax_rate": (syntax_pass / total_attempts) if total_attempts else None,
    })

row = {
    "cell_id": "${cell_id}",
    "dataset": "${dataset}",
    "status": "${status}",
    "returncode": ${rc},
    "sweep": "${sweep}",
    "max_steps": ${max_steps},
    "synthesis_iterations": ${iterations},
    "generation_model": "${GENERATION_MODEL}",
    "generation_backend": "${GENERATION_BACKEND}",
    "eval_model": "${EVAL_MODEL}",
    "eval_backend": "${EVAL_BACKEND}",
    "run_name": "${cell_run_name}",
    "log_path": "${log_path}",
    "started_at": "${started_at}",
    "finished_at": "${finished_at}",
}
summary_name = (
    f"{cell_run_name}_run_all_summary.json"
    if dataset == "gsm"
    else f"{cell_run_name}_summary.json"
)
summary_path = output_dir / "benchmarks" / summary_name
row["workflow_summary_path"] = str(summary_path)
summary = load_json(summary_path)
row["workflow_summary_found"] = isinstance(summary, dict)
if isinstance(summary, dict):
    if dataset == "gsm":
        add_gsm_metrics(row, summary)
    elif dataset == "spider":
        add_spider_metrics(row, summary)
    elif dataset == "smiles":
        add_smiles_metrics(row, summary)
with open("${SUMMARY_PATH}", "a", encoding="utf-8") as f:
    f.write(json.dumps(row) + "\\n")
PY
  echo "[ablation-grid:end] ${cell_id} rc=${rc} ${finished_at}"
}

run_maxsteps_sweep() {
  for dataset in "${DATASET_VALUES[@]}"; do
    for max_steps in "${MAX_STEPS_VALUES[@]}"; do
      run_cell "${dataset}" "${max_steps}" "${FIXED_SYNTHESIS_ITERATIONS}" "maxsteps"
    done
  done
}

run_iterations_sweep() {
  for dataset in "${DATASET_VALUES[@]}"; do
    for iterations in "${SYNTHESIS_ITERATIONS_VALUES[@]}"; do
      run_cell "${dataset}" "${FIXED_MAX_STEPS}" "${iterations}" "iterations"
    done
  done
}

case "${ABLATION_SWEEP}" in
  maxsteps|steps)
    run_maxsteps_sweep
    ;;
  iterations|iters)
    run_iterations_sweep
    ;;
  both)
    run_maxsteps_sweep
    run_iterations_sweep
    ;;
  *)
    echo "[ablation-grid:error] ABLATION_SWEEP must be maxsteps, iterations, or both; got ${ABLATION_SWEEP}" >&2
    exit 2
    ;;
esac

echo "[ablation-grid] complete summary=${SUMMARY_PATH}"
