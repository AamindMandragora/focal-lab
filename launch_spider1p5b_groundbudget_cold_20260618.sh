#!/bin/bash
# Spider-1.5B clean-win campaign — COLD START, MASK ON. Successor to
# launch_spider1p5b_prune_cold_20260618.sh (that run plateaued at the ~40% wall:
# att1 39.0/77.7, att2 15.3/24.7, att3 39.7/80.3, att4 0.3/0.0, att5 BEST 44.0/77.0;
# killed at att6 ex103/300 per the cancel-uninformative-runs rule — the menu prune
# did NOT move the ceiling).
#
# WHAT CHANGED vs the prune run (the ONE framework diff being tested here):
# the grounding helper RegenerateUnitOnGroundingFailure was BUDGET-REPARAMETERIZED.
# Before: its 6th arg was `maxStepsPerUnit` and its postcondition promised a PRODUCT
# bound `(maxRetries+1)*maxStepsPerUnit`. To satisfy the strategy template's
# `cost <= maxSteps` a caller had to prove `(maxRetries+1)*maxStepsPerUnit <= maxSteps`
# — nonlinear arithmetic Dafny's SMT solver won't auto-discharge — so any strategy
# calling it FAILED to verify, became non-evaluable, and the refinement beam (size 2)
# always dropped the helper-using sibling. Across all 6 attempts of the prune run the
# helper was NEVER called (0 call-sites in run.log).
# Now: the 6th arg is a single flat `budget: nat`; the postcondition promises
# `cost <= old(cost) + budget` (LINEAR), mirroring the already-verified
# CloseSpanWithinBudget. A caller composes it trivially (like CraneGeneration):
#   helpers.RegenerateUnitOnGroundingFailure(lm,parser,prompt,gen,eosToken,maxSteps,3,3)
# -> cost <= maxSteps with no product / no division lemma. This is MORE IterGen-faithful
# (IterGen caps total backtracking too). FAIR: a helper-library reparam, not a
# grammar/grader/split/eval-semantics change; no strategy/task guidance added.
#   lib synthesis/verify/library/VerifiedAgentSynthesis.dfy md5 cd3591cef10f8e4ce62ac36619c5e8c6
#     (was 7a611864...); 175 verified, 0 errors (Dafny 4.11.0).
#   prompts.py md5 65bffe893730daafebb937198fc5a20b (was dfdba5a6...): menu call-shape
#     `...eosToken, budget, maxRetries, maxRollbackBudget` + "Cost: bounded by `budget`".
#     The menu prune (universe 63) + IterGen-promotion fairfix are carried forward.
#   TDD red->green: a GeneratedCSD-template strategy calling the helper the natural way
#     FAILED under the old product-bound lib (cost + length postconditions, 2 errors)
#     and VERIFIES under the reparam'd lib (0 errors). Fairness 12/12, grounding 8/8.
#
# WATCH: (1) does the COLD author now CALL RegenerateUnitOnGroundingFailure (grep
# run.log for call-sites + grounded_units/grounded_True > 0), and (2) does adoption
# move accuracy off the 40% wall? Honest caveat: provable != adopted != a win —
# out-of-schema is a MINORITY failure mode; span-timing/closure dominate.
#
# COLD start (no --initial-strategy-file). Author = Bedrock claude-sonnet-4-6,
# thinking high (WORK cred AWS_BEARER_TOKEN_BEDROCK from synthesis/.env, us-east-1
# — NOT personal). Eval = Qwen2.5-1.5B via vLLM on GPU 1, 0.20 util.
# WIN BAR: beat IterGen 52.0%/94.7% -> >=157/300 accuracy, syntax within ~10-15pp.
set -u
cd /home/aadivyar/csd-generation
export SPIDER_DB_DIR=/home/aadivyar/csd-generation/synthesis/evaluate/syncode/syncode/utils/sql_spider_eval/databases
export CSD_API_MAX_RETRIES=10
export CSD_RECURRENCE_PENALTY=0.3
OUT=outputs/generated/spider1p5b_groundbudget_cold_20260618
mkdir -p "$OUT"
SPIDER_SPLIT=/home/aadivyar/csd-generation/environment/benchmark_splits/spider_dev_proportional_300x300_seed334.json

CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:${LD_LIBRARY_PATH:-} /apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
  --task 'Generate a single valid SQL query as exactly `SQL: <<YOUR QUERY>>`, using only the provided schema context.' \
  --dataset spider \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --eval-model Qwen/Qwen2.5-1.5B-Instruct --eval-backend vllm \
  --max-iterations 20 \
  --output-name spider1p5b_groundbudget_cold_20260618 \
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
echo "DONE_SPIDER1P5B_GROUNDBUDGET_COLD $(date)" | tee -a "$OUT/run.log"
