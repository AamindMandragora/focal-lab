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

GENERATION_MODEL="${GENERATION_MODEL:-gpt-5.4}"
GENERATION_BACKEND="${GENERATION_BACKEND:-openai}"
EVAL_MODEL="${EVAL_MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}"
EVAL_BACKEND="${EVAL_BACKEND:-vllm}"
DEVICE="${DEVICE:-cuda}"
CRANE_REPO="${CRANE_REPO:-/home/aadivyar/CRANE}"
CRANE_DEVICE="${CRANE_DEVICE:-cuda:0}"

GSM_SPLIT_FILE="${GSM_SPLIT_FILE:-${OUTPUT_DIR}/splits/gsm_absolute_rubric_seed123_train50_eval50.json}"
GSM_SOURCE_DIR="${GSM_SOURCE_DIR:-/home/aadivyar/CRANE/src/gsm_symbolic}"
SPIDER_SPLIT_FILE="${SPIDER_SPLIT_FILE:-${OUTPUT_DIR}/splits/spider_seed123_train50_test100.json}"
SPIDER_SOURCE="${SPIDER_SOURCE:-local}"
SMILES_CLASSES="${SMILES_CLASSES:-acrylates,chain_extenders,isocyanates}"

echo "[ablation-grid] run_name=${RUN_NAME}"
echo "[ablation-grid] datasets=${DATASET_VALUES[*]}"
echo "[ablation-grid] max_steps=${MAX_STEPS_VALUES[*]}"
echo "[ablation-grid] synthesis_iterations=${SYNTHESIS_ITERATIONS_VALUES[*]}"
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

kill_vllm_workers "before ablation grid"

run_cell() {
  local dataset="$1"
  local max_steps="$2"
  local iterations="$3"
  local cell_id="${dataset}_ablation_steps${max_steps}_iters${iterations}"
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
        --crane-repo "${CRANE_REPO}"
        --crane-device "${CRANE_DEVICE}"
        --cars-repo /home/aadivyar/cars
        --cars-style cars
        --cars-max-attempts-per-example 2000
        --cars-cuda-visible-devices "${CARS_CUDA_VISIBLE_DEVICES:-1,3}")
      ;;
    spider|sql)
      dataset="spider"
      cell_id="spider_ablation_steps${max_steps}_iters${iterations}"
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
        --cars-cuda-visible-devices "${CARS_CUDA_VISIBLE_DEVICES:-1,3}")
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
	        --cuda-visible-devices "${CARS_CUDA_VISIBLE_DEVICES:-1,3}"
	        --crane-repo "${CRANE_REPO}"
	        --crane-device "${CRANE_DEVICE}"
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
  kill_vllm_workers "before ${cell_id}"
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
row = {
    "cell_id": "${cell_id}",
    "dataset": "${dataset}",
    "status": "${status}",
    "returncode": ${rc},
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
with open("${SUMMARY_PATH}", "a", encoding="utf-8") as f:
    f.write(json.dumps(row) + "\\n")
PY
  echo "[ablation-grid:end] ${cell_id} rc=${rc} ${finished_at}"
}

for dataset in "${DATASET_VALUES[@]}"; do
  for max_steps in "${MAX_STEPS_VALUES[@]}"; do
    for iterations in "${SYNTHESIS_ITERATIONS_VALUES[@]}"; do
      run_cell "${dataset}" "${max_steps}" "${iterations}"
    done
  done
done

echo "[ablation-grid] complete summary=${SUMMARY_PATH}"
