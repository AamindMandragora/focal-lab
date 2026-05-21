#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/csd-generation"
if [[ -f synthesis/.env ]]; then set -a; source synthesis/.env; set +a; fi
export CONDA_PREFIX="/apps/conda/advayth2/envs/advayth2"
export PATH="/apps/conda/advayth2/envs/advayth2/bin:${PATH:-}"
[[ -d "/apps/conda/advayth2/envs/advayth2/lib" ]] && export LD_LIBRARY_PATH="/apps/conda/advayth2/envs/advayth2/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=3
export VAS_MAX_CUDA_DEVICES=1
export VAS_VLLM_GPU_MEMORY_UTILIZATION=0.65
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
LOG=/home/aadivyar/csd-generation/logs/tmux/full_matrix_20260520_100516.log
echo "Started:     $(date -Is)"                                                  | tee -a "$LOG"
echo "python:      $(which python)"                                              | tee -a "$LOG"
echo "GPU:         CUDA_VISIBLE_DEVICES=3 (util=0.65)"                          | tee -a "$LOG"
echo "Author:      claude-opus-4-7 via anthropic backend (default per opus4.7 profile)" | tee -a "$LOG"
echo "Models:      1.5B-Instruct + 7B-Coder-Instruct"                            | tee -a "$LOG"
echo "Cache mode:  --reuse-baselines (skip 11 paper-cached cells)"               | tee -a "$LOG"
echo "Quota hardstop: enabled (QUOTA_RE aborts on credit exhaustion)"            | tee -a "$LOG"
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
