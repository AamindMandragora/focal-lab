#!/usr/bin/env bash
set -euo pipefail

cd /home/aadivyar/csd-generation
export PYTHONPATH="/home/aadivyar/csd-generation:${PYTHONPATH:-}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-master_experiments_${RUN_STAMP}}"
OUTPUT_DIR="/home/aadivyar/csd-generation/outputs/generated-csd"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

cmd=(/opt/anaconda/bin/python scripts/master_experiment_matrix.py
  --run-name "${RUN_NAME}" \
  --output-dir "${OUTPUT_DIR}" \
  --models "${MODELS:-all}" \
  --datasets "${DATASETS:-all}" \
  --methods "${METHODS:-all}")

if [[ "${INCLUDE_ABLATIONS:-true}" == "false" ]]; then
  cmd+=(--no-include-ablations)
fi
if [[ "${INCLUDE_LOTTERY_ABLATION:-false}" == "true" ]]; then
  cmd+=(--include-lottery-ablation)
fi

exec "${cmd[@]}" "$@"
