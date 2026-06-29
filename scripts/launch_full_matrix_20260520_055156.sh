#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/csd-generation"
if [[ -f synthesis/.env ]]; then
  set -a
  # shellcheck disable=SC1091
  source synthesis/.env
  set +a
fi
# advayth2 shared conda env activation
export CONDA_PREFIX="/apps/conda/advayth2/envs/advayth2"
export PATH="/apps/conda/advayth2/envs/advayth2/bin:${PATH:-}"
if [[ -d "/apps/conda/advayth2/envs/advayth2/lib" ]]; then
  export LD_LIBRARY_PATH="/apps/conda/advayth2/envs/advayth2/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
# GPU + vllm pinning
export CUDA_VISIBLE_DEVICES=0
export VAS_MAX_CUDA_DEVICES=1
export VAS_VLLM_GPU_MEMORY_UTILIZATION=0.65
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
LOG=/home/aadivyar/csd-generation/logs/tmux/full_matrix_20260520_055156.log
echo "Started:     $(date -Is)"                                       | tee -a "$LOG"
echo "python:      $(which python)"                                   | tee -a "$LOG"
echo "CUDA dev:    $CUDA_VISIBLE_DEVICES (vllm util=$VAS_VLLM_GPU_MEMORY_UTILIZATION)" | tee -a "$LOG"
echo "Models:      Qwen2.5-1.5B-Instruct + Qwen2.5-Coder-7B-Instruct (14B + Llama skipped)" | tee -a "$LOG"
echo "Strategies:  defaults (unconstrained,gcd,crane,itergen,cars,metadecode) + ablations" | tee -a "$LOG"
echo "----"                                                            | tee -a "$LOG"
set +e
python run_all_tests.py \
  --models "Qwen/Qwen2.5-1.5B-Instruct,Qwen/Qwen2.5-Coder-7B-Instruct" \
  2>&1 | tee -a "$LOG"
EXIT=${PIPESTATUS[0]}
echo "----"                                                            | tee -a "$LOG"
echo "Exit:        $EXIT"                                              | tee -a "$LOG"
echo "Finished:    $(date -Is)"                                        | tee -a "$LOG"
