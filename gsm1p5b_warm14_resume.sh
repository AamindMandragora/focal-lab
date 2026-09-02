#!/bin/bash
# GSM-1.5B restart after the warm3 run exhausted 20 attempts at 42.9% acc / 59.2% syn.
# Diagnosis (saved-results/gsm1p5b-syntax-wall-diagnosis.md): model-written << spans in the
# FREE phase carry {var} placeholders the grammar rejects (no masking there), while the forced
# span opens at token ~860 where the model immediately EOSes (41/49 abandoned).
# Changes vs warm3:
#   - task description adds CSD-MECHANISM guidance (fair per project rules: how to form CSDs,
#     not how to solve math): route model-emitted << through the constrained path, force early,
#     enforce in-loop span closure.
#   - warm start from attempt 14's actual stored code (42.9/59.2).
#   - min-accuracy lowered 0.41 -> 0.33: bank the win at the TRUE bars (0.32 acc / 0.90 syn).
set -uo pipefail
cd /home/aadivyar/csd-generation
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/opt/anaconda/lib:${LD_LIBRARY_PATH:-}
NAME="gsm1p5b_seed123_warm14_20260610"

echo "=== GSM1P5B WARM14 RESUME START $(date) (mechanism guidance, bars 0.33/0.90) ==="
python -m synthesis.run_synthesis \
  --task "Solve math word problems step by step, wrapping intermediate symbolic expressions and the final answer inside << >> delimiters. CSD mechanics that matter: the eval scores EVERY visible << >> span against the grammar, including spans the model writes on its own during unconstrained generation - so the strategy must route ANY model-emitted '<<' into the parser-guided constrained path (an unconstrained chunk that stops at '<<' achieves this) rather than letting spans pass through unmasked. Open a forced constrained span early if the model has not produced one, and enforce the constrained-span token budget inside the loop so every opened span is closed at a grammar-complete point before the step budget runs out." \
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
  --adaptive-helper-mask --helper-selection-policy bandit \
  --refinement-beam-size 2 \
  --gsm-split-file environment/benchmark_splits/gsm_symbolic_crane_proportional_49x49_seed123.json \
  --gsm-split-name train \
  --initial-strategy-file /home/aadivyar/csd-generation/warmstart_gsm1p5b_attempt14.dfy
echo "EXIT_SYNTH_GSM1P5B_WARM14=$?"
echo "DONE_GSM1P5B_WARM14 $(date)"
