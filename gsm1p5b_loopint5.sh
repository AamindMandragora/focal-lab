#!/bin/bash
# GSM-1.5B loopint5: COLD-START suppress-then-force after loopint4 showed the warm anchor
# fights the mechanism switch. loopint4 (warm from loopint3-att7, an intercept strategy):
# 9 scored attempts, ZERO rationales adopted suppress-then-force - every attempt refined the
# anchor's "detect '<<' then constrain" architecture instead (anchored refinement bias).
# Scores drifted down (28.6/34.7 by att9). Fix here: (1) NO warm start - the author must
# design from the task text; (2) task text now opens with an explicit imperative ban on the
# intercept approach (CSD-mechanism guidance, fair per project rules - no task-specific or
# helper-name guidance added).
set -uo pipefail
cd /home/aadivyar/csd-generation
export PYTHONPATH=/home/aadivyar/csd-generation:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=1  # GPU 0 blocked: orphaned engine (12.7GB, ours, needs user-approved kill) + labmate job
export LD_LIBRARY_PATH=/opt/anaconda/lib:${LD_LIBRARY_PATH:-}
# survive Bedrock daily-quota windows: retries are read per-call from env
export CSD_API_MAX_RETRIES=10
NAME="gsm1p5b_seed123_loopint5_20260611"

echo "=== GSM1P5B-LOOPINT5 START $(date) (COLD start, suppress-then-force mandated, bars 0.33acc/0.90syn) ==="
python -m synthesis.run_synthesis \
  --task "Solve math word problems step by step, wrapping the final answer inside << >> delimiters. MANDATORY MECHANISM - read carefully: do NOT build a strategy that waits for the model to emit '<<' and then intercepts it. That reactive-interception architecture has been tested exhaustively (60+ attempts across prior runs) and cannot pass: token-level triggers miss the delimiter when the tokenizer splits it, the model opens several spans with symbolic variable names the grammar rejects, and intercepting an intermediate reasoning span burns the step budget mid-sentence. Every strategy you write for this task MUST instead use SUPPRESS-THEN-FORCE: (1) during the free reasoning phase, mask out the token(s) that begin the span-opening delimiter so the model CANNOT open a span on its own - it reasons in plain text instead; (2) after the reasoning phase, FORCE exactly one constrained span for the final answer - append the opening delimiter directly, drive the content through the parser-guided constrained path, and close at a grammar-complete point - reserving enough of the step budget for this span so it always completes; (3) never exit the span without emitting its closing delimiter (roll back to the last grammar-complete prefix and close there if generation stalls). The eval scores EVERY visible << >> span against the grammar, so one forced, fully parser-guided span per example - and zero model-opened spans - satisfies the delimiter requirement with grammar-valid content." \
  --dataset gsm_symbolic \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --anthropic-thinking enabled --anthropic-effort high --anthropic-thinking-display summarized \
  --eval-model Qwen/Qwen2.5-1.5B-Instruct --eval-backend vllm \
  --max-iterations 20 \
  --output-name "$NAME" --output-dir "outputs/generated/$NAME" \
  --min-accuracy 0.33 --min-syntax-rate 0.90 \
  --eval-sample-size 49 --eval-max-steps 900 --eval-step-token-budget 1 \
  --eval-max-seconds-per-example 120 \
  --eval-min-examples-before-threshold-stop 49 \
  --max-tokens 32768 \
  --restart-after-stuck-iters 0 \
  --vllm-gpu-memory-utilization 0.30 --device auto --vllm-tensor-parallel-size 1 \
  --no-adaptive-helper-mask \
  --refinement-beam-size 2 \
  --gsm-split-file environment/benchmark_splits/gsm_symbolic_crane_proportional_49x49_seed123.json \
  --gsm-split-name train
echo "EXIT_SYNTH_GSM1P5B_LOOPINT5=$?"
echo "DONE_GSM1P5B_LOOPINT5 $(date)"
