#!/bin/bash
# Spider-1.5B clean-win campaign — COLD START, MASK ON, with the deployed SymbolPosMap
# mid-query grounding RECOVERY path (2026-06-16). Successor to
# launch_spider1p5b_grounding_cold_20260615.sh.
#
# What changed vs launch_spider1p5b_grounding_cold_20260615.sh:
#   * RECOVERY now works. The grounding helper RegenerateUnitOnGroundingFailure grounds at
#     the SCHEMA-SYMBOL boundary (IterGen's SymbolPosMap: table_ref/column_ref completion via
#     the new Dafny extern CompletedSchemaSymbolCount), not the whole-query boundary. Per-symbol
#     checkpoints make a rollback ~1 symbol, so the ×0.3 recurrence penalty accumulates within a
#     cheap retry budget and a bad table/column name actually FLIPS to an in-schema one.
#     Verified end-to-end on the fixture re-eval (v3): concert_singer penalized count 1->2->3 then
#     flips to 'concert' and grounds ('unit fully grounded' 0 -> 60+ vs the old whole-query boundary).
#     Deployed library md5s: VerifiedAgentSynthesis.dfy db666d67 (171 verified/0 err),
#     parser_utils.py 5c8bc1a8, model_utils.py 3dfabe32.
#   * Recurrence penalty: CUMULATIVE (locked in 2026-06-16). CSD_RECURRENCE_PENALTY=0.3 and
#     CSD_RECURRENCE_FLAT is NOT set -> ln(0.3)*count (compounds per retry). This is the default;
#     set it explicitly here for auditability. (Flat/IterGen-faithful ×0.3-once recovers only
#     low-gap names; cumulative also flips high-gap JOIN names. User ruled cumulative fair: same
#     KIND of signal, no gold labels / no execution feedback.)
#   * Output redirected to $OUT/run.log so the run can be monitored while detached.
#
# Everything else is byte-identical to the 2026-06-15 COLD config:
#   COLD start (no --initial-strategy-file). MASK ON (--adaptive-helper-mask
#   --helper-selection-policy bandit). Accept bars --min-accuracy 0.55 (clean margin over IterGen
#   0.52) --min-syntax-rate 0.85; --eval-min-examples-before-threshold-stop 300 so no small-N cut.
#   Author = Bedrock claude-sonnet-4-6, thinking high (WORK Bedrock cred AWS_BEARER_TOKEN_BEDROCK
#   from the project .env, AWS_REGION=us-east-1 — NOT a personal account). Eval = Qwen2.5-1.5B-Instruct
#   via vLLM on GPU 2, 0.20 util (~8GB; GPU2 has ~23GB free).
# WIN BAR: beat IterGen 52.0%/94.7% -> >=157/300 accuracy, syntax within ~10-15pp.
set -u
cd /home/aadivyar/csd-generation
export SPIDER_DB_DIR=/home/aadivyar/csd-generation/synthesis/evaluate/syncode/syncode/utils/sql_spider_eval/databases
export CSD_API_MAX_RETRIES=10
export CSD_RECURRENCE_PENALTY=0.3
OUT=outputs/generated/spider1p5b_symbolrecover_cold_20260616
mkdir -p "$OUT"
SPIDER_SPLIT=/home/aadivyar/csd-generation/environment/benchmark_splits/spider_dev_proportional_300x300_seed334.json

CUDA_VISIBLE_DEVICES=2 LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:${LD_LIBRARY_PATH:-} /apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
  --task 'Generate a single valid SQL query as exactly `SQL: <<YOUR QUERY>>`, using only the provided schema context.' \
  --dataset spider \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --eval-model Qwen/Qwen2.5-1.5B-Instruct --eval-backend vllm \
  --max-iterations 20 \
  --output-name spider1p5b_symbolrecover_cold_20260616 \
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
echo "DONE_SPIDER1P5B_SYMBOLRECOVER_COLD $(date)" | tee -a "$OUT/run.log"
