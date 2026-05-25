#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/csd-generation"
if [[ -f synthesis/.env ]]; then set -a; source synthesis/.env; set +a; fi
export CONDA_PREFIX="/apps/conda/advayth2/envs/advayth2"
export PATH="/apps/conda/advayth2/envs/advayth2/bin:${PATH:-}"
[[ -d "/apps/conda/advayth2/envs/advayth2/lib" ]] && export LD_LIBRARY_PATH="/apps/conda/advayth2/envs/advayth2/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=1
export VAS_VLLM_GPU_MEMORY_UTILIZATION=0.80
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1

TS="$(date +%Y%m%d_%H%M%S)"
LOGDIR="logs/thinking_effort_high_logio_${TS}"
mkdir -p "$LOGDIR/io"
export CSD_PROMPT_LOG_DIR="$LOGDIR/io"

echo "Started:  $(date -Is)"
echo "Purpose:  Test (1) raw I/O capture + (3) effort=high (not xhigh) on thinking-on Opus 4.7"
echo "          Falsifies 'thinking pipeline is broken' AND tests 'xhigh over-engineers' theory"
echo "GPU:      CUDA_VISIBLE_DEVICES=1 (util=0.80)"
echo "Logs:     $LOGDIR"
echo "Raw I/O:  $LOGDIR/io/prompt_io.jsonl"
echo "----"

python -m synthesis.run_synthesis \
  --task "Solve math word problems step by step, wrapping intermediate symbolic expressions and the final answer inside << >> delimiters." \
  --dataset gsm_symbolic \
  --generation-model claude-opus-4-7 \
  --generation-backend anthropic \
  --anthropic-thinking adaptive \
  --anthropic-effort high \
  --anthropic-thinking-display summarized \
  --restart-after-stuck-iters 0 \
  --eval-model "Qwen/Qwen2.5-Coder-7B-Instruct" \
  --eval-backend vllm \
  --max-iterations 5 \
  --min-accuracy 0.30 \
  --min-syntax-rate 0.50 \
  --eval-sample-size 20 \
  --eval-max-steps 600 \
  --eval-step-token-budget 1 \
  --eval-max-seconds-per-example 90 \
  --eval-min-examples-before-threshold-stop 15 \
  --helper-selection-policy bandit \
  --refinement-beam-size 2 \
  --vllm-gpu-memory-utilization 0.80 \
  --device auto \
  --output-dir outputs/generated \
  --output-name thinking_effort_high_logio_${TS} \
  2>&1 | tee "$LOGDIR/run.log"

EXIT=${PIPESTATUS[0]}
echo "Run exit: $EXIT" | tee "$LOGDIR/summary.log"
echo "Finished: $(date -Is)"
echo "Logs in:  $LOGDIR"
echo "Raw I/O records: $(wc -l < $LOGDIR/io/prompt_io.jsonl 2>/dev/null || echo 0)"
