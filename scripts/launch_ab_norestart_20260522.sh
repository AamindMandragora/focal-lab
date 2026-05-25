#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/csd-generation"
if [[ -f synthesis/.env ]]; then set -a; source synthesis/.env; set +a; fi
export CONDA_PREFIX="/apps/conda/advayth2/envs/advayth2"
export PATH="/apps/conda/advayth2/envs/advayth2/bin:${PATH:-}"
[[ -d "/apps/conda/advayth2/envs/advayth2/lib" ]] && export LD_LIBRARY_PATH="/apps/conda/advayth2/envs/advayth2/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=0
export VAS_VLLM_GPU_MEMORY_UTILIZATION=0.80
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1

TS="$(date +%Y%m%d_%H%M%S)"
LOGDIR="logs/ab_norestart_${TS}"
mkdir -p "$LOGDIR"

echo "Started:  $(date -Is)"
echo "Purpose:  A/B test — does removing restart help extended-thinking Opus?"
echo "          Run A: thinking ON + restart OFF (--restart-after-stuck-iters 0)"
echo "          Run B: thinking OFF + restart default=2 (control)"
echo "GPU:      CUDA_VISIBLE_DEVICES=0 (util=0.80)"
echo "Logs:     $LOGDIR"
echo "----"

COMMON_ARGS=(
  --task "Solve math word problems step by step, wrapping intermediate symbolic expressions and the final answer inside << >> delimiters."
  --dataset gsm_symbolic
  --generation-model claude-opus-4-7
  --generation-backend anthropic
  --eval-model "Qwen/Qwen2.5-Coder-7B-Instruct"
  --eval-backend vllm
  --max-iterations 5
  --min-accuracy 0.30
  --min-syntax-rate 0.50
  --eval-sample-size 20
  --eval-max-steps 600
  --eval-step-token-budget 1
  --eval-max-seconds-per-example 90
  --eval-min-examples-before-threshold-stop 15
  --helper-selection-policy bandit
  --refinement-beam-size 2
  --vllm-gpu-memory-utilization 0.80
  --device auto
  --output-dir outputs/generated
)

echo "=== Run A: thinking ON + restart OFF ==="
set +e
python -m synthesis.run_synthesis \
  "${COMMON_ARGS[@]}" \
  --anthropic-thinking adaptive \
  --anthropic-effort xhigh \
  --anthropic-thinking-display summarized \
  --restart-after-stuck-iters 0 \
  --output-name ab_norestart_thinking_on_${TS} \
  2>&1 | tee "$LOGDIR/run_a_thinking_on_norestart.log"
RUN_A_EXIT=${PIPESTATUS[0]}
echo "Run A exit: $RUN_A_EXIT" | tee -a "$LOGDIR/summary.log"

echo "----"
echo "=== Run B: thinking OFF + restart default (control) ==="
python -m synthesis.run_synthesis \
  "${COMMON_ARGS[@]}" \
  --anthropic-thinking off \
  --output-name ab_norestart_thinking_off_${TS} \
  2>&1 | tee "$LOGDIR/run_b_thinking_off.log"
RUN_B_EXIT=${PIPESTATUS[0]}
echo "Run B exit: $RUN_B_EXIT" | tee -a "$LOGDIR/summary.log"
set -e

echo "----"
echo "Finished: $(date -Is)"
echo "Logs in:  $LOGDIR"
