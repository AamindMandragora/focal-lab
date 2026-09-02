#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/csd-generation"
if [[ -f synthesis/.env ]]; then
  set -a
  # shellcheck disable=SC1091
  source synthesis/.env
  set +a
fi
export CUDA_VISIBLE_DEVICES=0
export VAS_MAX_CUDA_DEVICES=1
export VAS_VLLM_GPU_MEMORY_UTILIZATION=0.65
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
LOGFILE_INNER="/home/aadivyar/csd-generation/logs/tmux/full_matrix_20260520_024953.log"
echo "Started: $(date -Is)" | tee -a "$LOGFILE_INNER"
echo "GPU 0 pinned, vllm util=0.65" | tee -a "$LOGFILE_INNER"
echo "Models: Qwen/Qwen2.5-1.5B-Instruct, Qwen/Qwen2.5-Coder-7B-Instruct (14B + Llama skipped)" | tee -a "$LOGFILE_INNER"
echo "Strategies: all defaults (unconstrained,gcd,crane,itergen,cars,metadecode) + ablations" | tee -a "$LOGFILE_INNER"
echo "Command:" | tee -a "$LOGFILE_INNER"
echo "  python run_all_tests.py --models Qwen/Qwen2.5-1.5B-Instruct,Qwen/Qwen2.5-Coder-7B-Instruct" | tee -a "$LOGFILE_INNER"
echo "----" | tee -a "$LOGFILE_INNER"
set +e
python run_all_tests.py \
  --models "Qwen/Qwen2.5-1.5B-Instruct,Qwen/Qwen2.5-Coder-7B-Instruct" \
  2>&1 | tee -a "$LOGFILE_INNER"
EXIT=${PIPESTATUS[0]}
echo "----" | tee -a "$LOGFILE_INNER"
echo "Exit: $EXIT" | tee -a "$LOGFILE_INNER"
echo "Finished: $(date -Is)" | tee -a "$LOGFILE_INNER"
