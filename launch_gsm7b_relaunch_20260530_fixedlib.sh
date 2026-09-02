#!/bin/bash
# GSM-7B relaunch 2026-05-30 — FIXED-LIBRARY run, TRUE published bar.
# Same synthesis config as launch_gsm7b_newprompt.sh EXCEPT:
#   1. Runs on the fixed CloseConstrainedSpan library (rendered-text-suffix dedup,
#      RenderedEndsWith) already in synthesis/verify/library/VerifiedAgentSynthesis.dfy
#      — removes the doubled-'>>' syntax cap that limited prior 7B strategies.
#   2. min-accuracy raised 0.25 -> 0.40 so a harness "success" means a REAL win:
#      strictly beat published CRANE-7B (Qwen2.5-Coder-7B) = 39% acc / 94% syntax.
#      (20/50 = 0.40 > 0.39.) min-syntax 0.90 holds the comparable-syntax requirement.
#   3. max-iterations 10 -> 15 (harder target: best prior is 30%/92%, a ~9pp accuracy
#      gap; give the author more attempts to evolve the inner math, not just syntax).
#   4. GPU 2 -> GPU 1 (static co-tenant, 21.9GB free) and util 0.55 -> 0.45 (~18GB)
#      to fit safely alongside the neighbor without OOM. Eval is token-budget-1 /
#      sequential, so a smaller KV cache is fine.
# New output dir/name/log. Nothing else changed (task string, prompt, splits identical).
set -u
cd /home/aadivyar/csd-generation
OUT=outputs/generated/ralph_7B_gsm_relaunch_20260530_fixedlib
mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:$LD_LIBRARY_PATH nohup /apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
  --task 'Solve math word problems step by step, wrapping intermediate symbolic expressions and the final answer inside << >> delimiters.' \
  --dataset gsm_symbolic \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --eval-model Qwen/Qwen2.5-Coder-7B-Instruct --eval-backend vllm \
  --max-iterations 15 \
  --output-name ralph_7B_gsm_relaunch_20260530_fixedlib \
  --min-accuracy 0.40 --min-syntax-rate 0.90 \
  --eval-sample-size 50 --eval-max-steps 900 --eval-step-token-budget 1 \
  --eval-max-seconds-per-example 90 --eval-min-examples-before-threshold-stop 50 \
  --max-tokens 32768 --restart-after-stuck-iters 0 \
  --vllm-gpu-memory-utilization 0.45 --device auto \
  --output-dir "$OUT" \
  --adaptive-helper-mask --helper-selection-policy bandit --refinement-beam-size 2 \
  --anthropic-thinking enabled --anthropic-effort high --anthropic-thinking-display summarized \
  --vllm-tensor-parallel-size 1 \
  --gsm-split-file /home/aadivyar/csd-generation/environment/benchmark_splits/gsm_symbolic_crane_proportional.json \
  --gsm-split-name eval \
  > /tmp/ralph_7B_gsm_relaunch_20260530_fixedlib.log 2>&1 &
echo "GSM-7B fixedlib pid: $!  -> $OUT  log=/tmp/ralph_7B_gsm_relaunch_20260530_fixedlib.log"
