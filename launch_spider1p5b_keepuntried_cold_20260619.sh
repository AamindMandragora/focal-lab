#!/bin/bash
# Spider-1.5B grounding-helper SURFACING campaign — COLD START, MASK ON.
# Successor to launch_spider1p5b_authorbody_cold_20260619.sh.
#
# WHAT CHANGED vs the authorbody run (the framework diff being tested here):
# the BANDIT KEEP-ALL-UNTRIED fix in synthesis/evaluate/feedback_loop.py
# (_compute_allowed_helpers_bandit, new md5 481b2110). The old mask kept only the
# single alphabetically-first untried helper per iteration (explore_untried=1),
# which hid ~56/57 untried helpers — including RegenerateUnitOnGroundingFailure
# (the IterGen-like grounding/unit-rewind helper) — by alphabetical position with
# zero evidence against them. The fix keeps EVERY zero-pull helper on the menu
# each iteration; the mask still prunes helpers that HAVE been tried and did worse.
# Fair: a policy fix to OUR pipeline's mask, not grammar/grader/split/eval-semantics;
# mask still ON. TDD red->green on focal (tests/test_helper_bandit_keep_untried.py,
# 2 passed: 56 untried pruned -> 0 untried pruned). User ruling 2026-06-19.
#   NOT changed: grammar, grader, splits, recurrence penalty, eval semantics,
#   the Dafny library (lib md5 3c088b24), the Pattern A template (md5 41c272a1,
#   still live). No warm start.
#
# ULTIMATE GOAL (user 2026-06-19): surface RegenerateUnitOnGroundingFailure on the
# author's menu and get DATA on whether the author adopts it and whether it works.
# Confirmed pre-flight: that helper IS in the author universe (63 helpers), IS
# prunable, and was being masked out after warm-up under the old policy.
#
# WATCH: does the cold author now WRITE helpers.RegenerateUnitOnGroundingFailure
# into a strategy (it is now on the menu every iteration), and what acc/syntax
# does that strategy get?
#
# COLD start (no --initial-strategy-file). Author = Bedrock claude-sonnet-4-6,
# thinking high (WORK cred AWS_BEARER_TOKEN_BEDROCK from synthesis/.env, us-east-1
# — NOT personal; approved by user 2026-06-19). Eval = Qwen2.5-1.5B via vLLM on
# GPU 1, 0.20 util.
# WIN BAR: beat IterGen 52.0%/94.7% -> >=157/300 accuracy, syntax within ~10-15pp.
set -u
cd /home/aadivyar/csd-generation
export SPIDER_DB_DIR=/home/aadivyar/csd-generation/synthesis/evaluate/syncode/syncode/utils/sql_spider_eval/databases
export CSD_API_MAX_RETRIES=10
export CSD_RECURRENCE_PENALTY=0.3
OUT=outputs/generated/spider1p5b_keepuntried_cold_20260619
mkdir -p "$OUT"
SPIDER_SPLIT=/home/aadivyar/csd-generation/environment/benchmark_splits/spider_dev_proportional_300x300_seed334.json

CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:${LD_LIBRARY_PATH:-} /apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
  --task 'Generate a single valid SQL query as exactly `SQL: <<YOUR QUERY>>`, using only the provided schema context.' \
  --dataset spider \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --eval-model Qwen/Qwen2.5-1.5B-Instruct --eval-backend vllm \
  --max-iterations 20 \
  --output-name spider1p5b_keepuntried_cold_20260619 \
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
echo "DONE_SPIDER1P5B_KEEPUNTRIED_COLD $(date)" | tee -a "$OUT/run.log"
