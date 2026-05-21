#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/csd-generation"
if [[ -f synthesis/.env ]]; then set -a; source synthesis/.env; set +a; fi
export CONDA_PREFIX="/apps/conda/advayth2/envs/advayth2"
export PATH="/apps/conda/advayth2/envs/advayth2/bin:${PATH:-}"
[[ -d "/apps/conda/advayth2/envs/advayth2/lib" ]] && export LD_LIBRARY_PATH="/apps/conda/advayth2/envs/advayth2/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=2
export VAS_MAX_CUDA_DEVICES=1
export VAS_VLLM_GPU_MEMORY_UTILIZATION=0.85
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
LOG=/home/aadivyar/csd-generation/logs/tmux/full_matrix_7b_gpu2_20260520_100516.log
echo "Started:     $(date -Is)"                                                  | tee -a "$LOG"
echo "python:      $(which python)"                                              | tee -a "$LOG"
echo "GPU:         CUDA_VISIBLE_DEVICES=2 (util=0.85)"                          | tee -a "$LOG"
echo "Author:      claude-opus-4-7 via anthropic backend"                        | tee -a "$LOG"
echo "Scope:       7B-Coder-Instruct only, Phase 1 only (--skip-ablations)"     | tee -a "$LOG"
echo "----"                                                                      | tee -a "$LOG"
set +e
python run_all_tests.py \
  --models "Qwen/Qwen2.5-Coder-7B-Instruct" \
  --reuse-baselines \
  --skip-ablations \
  2>&1 | tee -a "$LOG"
EXIT=${PIPESTATUS[0]}
echo "----"                                                                      | tee -a "$LOG"
echo "Exit:        $EXIT"                                                        | tee -a "$LOG"
echo "Finished:    $(date -Is)"                                                  | tee -a "$LOG"
