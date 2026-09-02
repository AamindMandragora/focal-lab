#!/bin/bash
# Spider-1.5B clean-win campaign — COLD START, MASK ON, after the VERIFICATION-FRICTION FIX
# (2026-06-16). Successor to launch_spider1p5b_symbolrecover_cold_20260616.sh.
#
# WHY: the symbolrecover COLD run got NO win (best 42/86, below the 51/98 near-tie) and 8 of 19
# attempts (42%) died at the Dafny prover instead of the task — the author kept hand-rolling loops
# it couldn't prove against the template's |generated| <= |generatedPrefix| + maxSteps and
# cost <= maxSteps postconditions. User chose (2026-06-16) to reduce that friction, then relaunch.
#
# WHAT CHANGED (deployed to focal; this run picks them up, config is otherwise byte-identical to the
# symbolrecover run so the ONLY difference is the friction fix — clean controlled comparison):
#   1. VerifiedAgentSynthesis.dfy (md5 4d96bb8f, re-verified 171/0): added the missing length ensures
#      to RegenerateUnitOnCheckFailure so it matches its grounding sibling (was a guaranteed verify-fail
#      for any strategy using it).
#   2. feedback_loop.py (md5 1bb88897): moved the budget-carrying scaffolds RolloutConstrainedWithPenalties
#      and ConstrainedGeneration from PRUNABLE to NON_PRUNABLE (always-visible). The mask STAYS ON; this
#      only exempts two more verification-friendly scaffolds (like GenerateWithManagedSpan already is), so
#      the author can DELEGATE to a verifying scaffold for penalized / constrained-phase generation instead
#      of hand-rolling a loop and failing the budget proof.
#   NOT changed: grammar, grader, splits, prompts (no strategy guidance). Mask ON. Cumulative recurrence
#   penalty (CSD_RECURRENCE_PENALTY=0.3, no FLAT) kept identical to the symbolrecover config.
#
# WATCH (de-facto probe): the per-attempt verify-fail rate in the FIRST ~4-5 attempts. If it stays ~40%+
# the friction fix didn't help -> kill and reassess. If it drops, let it ride.
#
# COLD start (no --initial-strategy-file). Author = Bedrock claude-sonnet-4-6, thinking high (WORK cred
# AWS_BEARER_TOKEN_BEDROCK, us-east-1 — NOT personal). Eval = Qwen2.5-1.5B via vLLM on GPU 2, 0.20 util.
# WIN BAR: beat IterGen 52.0%/94.7% -> >=157/300 accuracy, syntax within ~10-15pp.
set -u
cd /home/aadivyar/csd-generation
export SPIDER_DB_DIR=/home/aadivyar/csd-generation/synthesis/evaluate/syncode/syncode/utils/sql_spider_eval/databases
export CSD_API_MAX_RETRIES=10
export CSD_RECURRENCE_PENALTY=0.3
OUT=outputs/generated/spider1p5b_frictionfix_cold_20260616
mkdir -p "$OUT"
SPIDER_SPLIT=/home/aadivyar/csd-generation/environment/benchmark_splits/spider_dev_proportional_300x300_seed334.json

CUDA_VISIBLE_DEVICES=2 LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:${LD_LIBRARY_PATH:-} /apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
  --task 'Generate a single valid SQL query as exactly `SQL: <<YOUR QUERY>>`, using only the provided schema context.' \
  --dataset spider \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --eval-model Qwen/Qwen2.5-1.5B-Instruct --eval-backend vllm \
  --max-iterations 20 \
  --output-name spider1p5b_frictionfix_cold_20260616 \
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
echo "DONE_SPIDER1P5B_FRICTIONFIX_COLD $(date)" | tee -a "$OUT/run.log"
