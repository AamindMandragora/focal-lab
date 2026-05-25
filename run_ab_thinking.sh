#!/bin/bash
# A/B test: extended thinking ON (adaptive xhigh summarized) vs OFF
# on claude-opus-4-7 author, GSM-Symbolic eval split, Qwen2.5-Coder-7B eval via vllm.
# Same --eval-seed and same --gsm-split so both runs see the same examples.

set -eo pipefail
cd /home/aadivyar/csd-generation

# Load ANTHROPIC_API_KEY from .env
set -a
source .env
set +a

LOG_DIR="logs/ab_thinking_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "Logging to $LOG_DIR" >&2

COMMON_ARGS=(
    --task "Solve math word problems step by step, wrapping intermediate symbolic expressions and the final answer inside << >> delimiters."
    --dataset gsm_symbolic
    --max-iterations 5
    --generation-model claude-opus-4-7
    --generation-backend anthropic
    --eval-model "Qwen/Qwen2.5-Coder-7B-Instruct"
    --eval-backend vllm
    --max-tokens 32768
    --device auto
    --min-accuracy 0.30
    --min-syntax-rate 0.90
    --eval-sample-size 20
    --eval-seed 42
    --eval-max-steps 600
    --eval-step-token-budget 1
    --eval-max-seconds-per-example 90
    --eval-min-examples-before-threshold-stop 15
    --helper-selection-policy bandit
    --refinement-beam-size 2
    --vllm-gpu-memory-utilization 0.80
    --vllm-tensor-parallel-size 1
    --gsm-split-file /home/aadivyar/csd-generation/environment/benchmark_splits/gsm_symbolic_crane_proportional.json
    --gsm-split-name eval
    --output-dir outputs/generated
)

echo "=== Run A: thinking ON (adaptive xhigh summarized) ==="
CUDA_VISIBLE_DEVICES=0 python -m synthesis.run_synthesis \
    "${COMMON_ARGS[@]}" \
    --anthropic-thinking adaptive \
    --anthropic-effort xhigh \
    --anthropic-thinking-display summarized \
    --output-name ab_thinking_on \
    2>&1 | tee "$LOG_DIR/run_a_thinking_on.log"
RUN_A_RC=${PIPESTATUS[0]}
echo "Run A exit: $RUN_A_RC" | tee -a "$LOG_DIR/summary.log"

echo ""
echo "=== Run B: thinking OFF ==="
CUDA_VISIBLE_DEVICES=0 python -m synthesis.run_synthesis \
    "${COMMON_ARGS[@]}" \
    --anthropic-thinking off \
    --output-name ab_thinking_off \
    2>&1 | tee "$LOG_DIR/run_b_thinking_off.log"
RUN_B_RC=${PIPESTATUS[0]}
echo "Run B exit: $RUN_B_RC" | tee -a "$LOG_DIR/summary.log"

echo ""
echo "=== A/B done; logs in $LOG_DIR ==="
echo "$LOG_DIR" > /tmp/ab_thinking.log_dir
