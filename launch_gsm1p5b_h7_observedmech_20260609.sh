#!/bin/bash
# H7: enrich the span-closure feedback hint with the OBSERVED failure mechanism.
# EXACTLY ONE variable vs the H6 launch: the return string of
#   synthesis/evaluate/feedback_loop.py::_span_not_closed_hint now also describes the
#   two patterns seen in the actual failing eval outputs -- (1) most unclosed spans
#   already hold a COMPLETE expression and lack only the closing `>>`, and (2) the step
#   budget is consumed by a non-progressing loop that re-selects the same rejected
#   token after a dead end. Mechanism-level / study-valid: describes CSD decoding
#   behavior (spans, budget, rollback, delimiters), NOT the GSM task; names NO library
#   helper and NO call sequence (discovery goal preserved). The H6 threshold (0.1) and
#   all prior confirmed corrections (H4 de-bias, H5 postcondition) are KEPT unchanged.
# Everything else identical to H6: warmstart baseline body, --min-syntax-rate 0.92
#   (forces iteration), --min-accuracy 0.31, --max-iterations 15, Sonnet-4-6 author
#   thinking-high, seed429 TRAIN.
set -uo pipefail
cd /home/aadivyar/csd-generation

# GPU 2 only: GPU 1 holds orphaned vLLM workers from the crashed H6, GPU 3 is another
# user, and advayth2's idle SMILES retry-loop targets device 0 when it wakes. The run is
# tensor-parallel-size 1, so a single free GPU (2) is all it needs and avoids all collisions.
export CUDA_VISIBLE_DEVICES=2
export LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:${LD_LIBRARY_PATH:-}

OUT=outputs/generated/ralph_1p5B_gsm_h7_observedmech_20260609
GSM_SPLIT=/home/aadivyar/csd-generation/environment/benchmark_splits/gsm_symbolic_crane_proportional_49x49_seed429.json
LOG=/tmp/ralph_1p5B_gsm_h7_observedmech_20260609.log

mkdir -p "$OUT"

nohup /apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
  --task 'Solve math word problems step by step, wrapping intermediate symbolic expressions and the final answer inside << >> delimiters.' \
  --dataset gsm_symbolic \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --anthropic-thinking enabled --anthropic-effort high --anthropic-thinking-display summarized \
  --eval-model Qwen/Qwen2.5-1.5B-Instruct --eval-backend vllm \
  --max-iterations 15 \
  --output-name ralph_1p5B_gsm_h7_observedmech_20260609 \
  --output-dir "$OUT" \
  --min-accuracy 0.31 --min-syntax-rate 0.92 \
  --eval-sample-size 49 --eval-max-steps 900 --eval-step-token-budget 1 \
  --eval-max-seconds-per-example 120 --eval-min-examples-before-threshold-stop 49 \
  --max-tokens 32768 --restart-after-stuck-iters 0 \
  --vllm-gpu-memory-utilization 0.18 --device auto \
  --vllm-tensor-parallel-size 1 \
  --adaptive-helper-mask --helper-selection-policy bandit --refinement-beam-size 2 \
  --gsm-split-file "$GSM_SPLIT" --gsm-split-name train \
  --initial-strategy-file /home/aadivyar/csd-generation/gsm1p5b_seed429_warmstart_body.dfy \
  > "$LOG" 2>&1 &

echo "PID=$!"
echo "LOG=$LOG"
echo "OUT=$OUT"
