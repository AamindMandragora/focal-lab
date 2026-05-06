#!/usr/bin/env bash
set -euo pipefail

cd /home/aadivyar/csd-generation
export PYTHONPATH="/home/aadivyar/csd-generation:${PYTHONPATH:-}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

TASK_DESC="Solve math word problems step by step, writing each arithmetic computation inside << >> delimiters."
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-gsm_ablation_grid_${RUN_STAMP}}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/aadivyar/csd-generation/outputs/generated-csd}"
LOG_DIR="${OUTPUT_DIR}/logs/${RUN_NAME}"
mkdir -p "${LOG_DIR}"

MAX_STEPS_VALUES=(${MAX_STEPS_VALUES:-256 512 1024})
SYNTHESIS_ITERATIONS_VALUES=(${SYNTHESIS_ITERATIONS_VALUES:-5 10 15 20})

GENERATION_MODEL="${GENERATION_MODEL:-gpt-5.4}"
GENERATION_BACKEND="${GENERATION_BACKEND:-openai}"
EVAL_MODEL="${EVAL_MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}"
EVAL_BACKEND="${EVAL_BACKEND:-vllm}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SUMMARY_PATH="${OUTPUT_DIR}/benchmarks/${RUN_NAME}_summary.jsonl"
mkdir -p "$(dirname "${SUMMARY_PATH}")"

echo "[ablation-grid] run_name=${RUN_NAME}"
echo "[ablation-grid] max_steps=${MAX_STEPS_VALUES[*]}"
echo "[ablation-grid] synthesis_iterations=${SYNTHESIS_ITERATIONS_VALUES[*]}"
echo "[ablation-grid] summary=${SUMMARY_PATH}"

for max_steps in "${MAX_STEPS_VALUES[@]}"; do
  for iterations in "${SYNTHESIS_ITERATIONS_VALUES[@]}"; do
    cell_id="gsm_ablation_steps${max_steps}_iters${iterations}"
    output_name="${RUN_NAME}_${cell_id}"
    log_path="${LOG_DIR}/${cell_id}.log"
    started_at="$(date --iso-8601=seconds)"
    echo "[ablation-grid:start] ${cell_id} ${started_at}"

    set +e
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" /opt/anaconda/bin/python run_synthesis.py \
      --task "${TASK_DESC}" \
      --dataset gsm_symbolic \
      --max-iterations "${iterations}" \
      --generation-model "${GENERATION_MODEL}" \
      --generation-backend "${GENERATION_BACKEND}" \
      --eval-model "${EVAL_MODEL}" \
      --eval-backend "${EVAL_BACKEND}" \
      --output-name "${output_name}" \
      --temperature 0.7 \
      --device cuda \
      --min-accuracy 0.30 \
      --min-syntax-rate 0.50 \
      --eval-sample-size 10 \
      --eval-seed 123 \
      --eval-max-steps "${max_steps}" \
      --eval-step-token-budget 1 \
      --vllm-tensor-parallel-size 1 \
      --vllm-max-model-len 4096 \
      --vllm-gpu-memory-utilization 0.40 \
      --synthesis-max-tokens 2048 \
      >"${log_path}" 2>&1
    rc=$?
    set -e

    finished_at="$(date --iso-8601=seconds)"
    status="completed"
    if [[ "${rc}" -ne 0 ]]; then
      status="failed"
    fi
    /opt/anaconda/bin/python - <<PY
import json
row = {
    "cell_id": "${cell_id}",
    "status": "${status}",
    "returncode": ${rc},
    "max_steps": ${max_steps},
    "synthesis_iterations": ${iterations},
    "generation_model": "${GENERATION_MODEL}",
    "generation_backend": "${GENERATION_BACKEND}",
    "eval_model": "${EVAL_MODEL}",
    "eval_backend": "${EVAL_BACKEND}",
    "output_name": "${output_name}",
    "log_path": "${log_path}",
    "started_at": "${started_at}",
    "finished_at": "${finished_at}",
}
with open("${SUMMARY_PATH}", "a", encoding="utf-8") as f:
    f.write(json.dumps(row) + "\\n")
PY
    echo "[ablation-grid:end] ${cell_id} rc=${rc} ${finished_at}"
  done
done

echo "[ablation-grid] complete summary=${SUMMARY_PATH}"
