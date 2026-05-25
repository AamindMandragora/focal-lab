#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/csd-generation"
if [[ -f synthesis/.env ]]; then set -a; source synthesis/.env; set +a; fi
export CONDA_PREFIX="/apps/conda/advayth2/envs/advayth2"
export PATH="/apps/conda/advayth2/envs/advayth2/bin:${PATH:-}"
if [[ -d "/apps/conda/advayth2/envs/advayth2/lib" ]]; then
  export LD_LIBRARY_PATH="/apps/conda/advayth2/envs/advayth2/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export CUDA_VISIBLE_DEVICES=0
export VAS_MAX_CUDA_DEVICES=1
export VAS_VLLM_GPU_MEMORY_UTILIZATION=0.65
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
LOG=/home/aadivyar/csd-generation/logs/tmux/full_matrix_20260520_074111.log
echo "Started:     $(date -Is)"                                                  | tee -a "$LOG"
echo "python:      $(which python)"                                              | tee -a "$LOG"
echo "Models:      Qwen2.5-1.5B-Instruct + Qwen2.5-Coder-7B-Instruct"            | tee -a "$LOG"
echo "Cache mode:  --reuse-baselines (skip cells with usable JSONs)"             | tee -a "$LOG"
echo "Pre-pop:     9 paper-cached baseline JSONs (CRANE T1 + IterGen T1)"        | tee -a "$LOG"
echo "----"                                                                      | tee -a "$LOG"
set +e
python run_all_tests.py \
  --models "Qwen/Qwen2.5-1.5B-Instruct,Qwen/Qwen2.5-Coder-7B-Instruct" \
  --reuse-baselines \
  2>&1 | tee -a "$LOG"
EXIT=${PIPESTATUS[0]}
echo "----"                                                                      | tee -a "$LOG"
echo "Exit:        $EXIT"                                                        | tee -a "$LOG"
echo "Finished:    $(date -Is)"                                                  | tee -a "$LOG"
