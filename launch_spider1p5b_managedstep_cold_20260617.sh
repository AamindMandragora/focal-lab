#!/bin/bash
# Spider-1.5B clean-win campaign — COLD START, MASK ON, with the new ManagedStep
# per-step self-discharging helper (2026-06-17). Successor to
# launch_spider1p5b_verbdiag_cold_20260617.sh.
#
# WHY: the verbdiag run (categorized verification remediation hints) did NOT cut the
# verify-fail rate — the PROGRESS postcondition still failed att2/3/4/8/9/10/11/12 WITH
# remediation present. Root cause: the low-level step helpers split "grow output" from
# "count cost" (AppendConstrainedToken charges 0; UnconstrainedStep/AdaptiveConstrainedStep
# charge +1 only on the non-EOS path), so hand-composing them across several guarded loops
# makes "exactly one counted step on every path" unprovable. att5 only recovered by
# collapsing to a single loop by hand.
#
# WHAT CHANGED (deployed + verified on focal; config otherwise byte-identical to verbdiag so
# the ONLY framework difference is the new helper — clean controlled comparison):
#   synthesis/verify/library/VerifiedAgentSynthesis.dfy (md5 7a611864df2709829d498b8225ab4f99):
#     added method ManagedStep — one iteration of GenerateWithManagedSpan's loop body, with a
#     UNIFORM per-step contract (cost == old(cost)+1, |grown| <= +1, well-formed span out,
#     `done` flag for EOS/closed). Full library re-verifies 175 verified / 0 errors. A single
#     `while steps < maxSteps` loop calling only ManagedStep + `cost := helpers.cost` discharges
#     length/cost/progress by construction (TDD probe: 2 verified / 0 errors).
#   synthesis/generate/prompts.py (md5 21d16d8679bc71e19ac1a3f6dba96bb2): one mechanism-only
#     bullet for helpers.ManagedStep in `## Available Tools` (same format as GenerateWithManagedSpan;
#     no usage/strategy/baseline hint). Scraped into the helper universe (size 72).
#   synthesis/evaluate/feedback_loop.py (md5 e9a976fea3e0e43ad8b56cd1b6ab6d36): "ManagedStep" added
#     to PRUNABLE_HELPERS so the bandit decides visibility (visible during the <3-attempt warm-up).
#   NOT changed: grammar, grader, splits, helper mask (still ON/bandit), recurrence penalty, the
#   low-level step helpers (kept, NOT hidden — user ruling 2026-06-17), eval semantics. No warm start.
#
# WATCH (the user's question): does adding ManagedStep make VERIFICATION EASIER — i.e. does the
# author adopt it, and does the verify-fail rate per attempt drop vs verbdiag (PROGRESS failed
# ~8/12 attempts there)? Secondary: does accuracy clear the win bar.
#
# COLD start (no --initial-strategy-file). Author = Bedrock claude-sonnet-4-6, thinking high (WORK
# cred AWS_BEARER_TOKEN_BEDROCK, us-east-1 — NOT personal). Eval = Qwen2.5-1.5B via vLLM on GPU 1,
# 0.20 util.
# WIN BAR: beat IterGen 52.0%/94.7% -> >=157/300 accuracy, syntax within ~10-15pp.
set -u
cd /home/aadivyar/csd-generation
export SPIDER_DB_DIR=/home/aadivyar/csd-generation/synthesis/evaluate/syncode/syncode/utils/sql_spider_eval/databases
export CSD_API_MAX_RETRIES=10
export CSD_RECURRENCE_PENALTY=0.3
OUT=outputs/generated/spider1p5b_managedstep_cold_20260617
mkdir -p "$OUT"
SPIDER_SPLIT=/home/aadivyar/csd-generation/environment/benchmark_splits/spider_dev_proportional_300x300_seed334.json

CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:${LD_LIBRARY_PATH:-} /apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
  --task 'Generate a single valid SQL query as exactly `SQL: <<YOUR QUERY>>`, using only the provided schema context.' \
  --dataset spider \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --eval-model Qwen/Qwen2.5-1.5B-Instruct --eval-backend vllm \
  --max-iterations 20 \
  --output-name spider1p5b_managedstep_cold_20260617 \
  --min-accuracy 0.55 --min-syntax-rate 0.85 \
  --eval-sample-size 300 --eval-max-steps 1200 --eval-step-token-budget 1 \
  --eval-max-seconds-per-example 300 --eval-min-examples-before-threshold-stop 300 \
  --max-tokens 32768 --restart-after-stuck-iters 0 \
  --vllm-gpu-memory-utilization 0.20 --device auto \
  --output-dir "$OUT" \
  --adaptive-helper-mask --helper-selection-policy bandit --refinement-beam-size 2 \
  --anthropic-thinking enabled --anthropic-effort high --anthropic-thinking-display summarized \
  --vllm-tensor-parallel-size 1 \
  --spider-split-file "$SPIDER_SPLIT" --spider-split-name train \
  > "$OUT/run.log" 2>&1
echo "SYNTH_EXIT=$?" | tee -a "$OUT/run.log"
echo "DONE_SPIDER1P5B_MANAGEDSTEP_COLD $(date)" | tee -a "$OUT/run.log"
