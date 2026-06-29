#!/bin/bash
# Spider-1.5B clean-win campaign — COLD START, MASK ON, span-close fix (2026-06-15) — RELAUNCH r2.
#
# WHY r2: the first span-close run (spider1p5b_spanclose_cold_20260615) was killed at att11/20. Best
# REAL (full-300) attempt was att1 39.3%/79.7% — well below the 0.55 bar — and the loop was anchoring
# on att8's "50.0%", which was an INFLATED truncated number (only 28 of 300 examples ran before a
# timeout early-stop). The author was chasing that mirage.
#
# WHAT CHANGED for r2 (framework only — no warm start, no new strategy guidance to the author):
#   * EARLY-STOP DENOMINATOR FIX is now live (deployed + verified on focal 2026-06-15):
#     synthesis/evaluate/metrics.py::choose_denominator_basis — on ANY early-stop, accuracy AND
#     syntax_rate are now reported over the full intended 300 (un-run examples counted as wrong/fail),
#     not over the truncated N that ran. So a timeout-truncated attempt now reports its HONEST low
#     accuracy and the author anchors on real full-300 results (e.g. att1-style), not an inflated 28/300.
#     Also: feedback_loop.py fallback-winner scan now skips early_stopped attempts.
#     This change is byte-identical for FULL runs, so baselines/accepted runs are unaffected (fair).
#   * Fresh output dir (no resume/append). COLD start — no --initial-strategy-file.
# EVERYTHING ELSE is identical to the killed run (CloseSpanWithinBudget helper + fixed/added few-shot
# examples + grounding helper retained, mask ON, same bars, same split, same GPU).
#
#   * MASK ON (2026-06-14 ruling): --adaptive-helper-mask --helper-selection-policy bandit.
#   * Accept bars: --min-accuracy 0.55 (clean margin over IterGen 0.52) --min-syntax-rate 0.85.
#   * GPU 1 (freed after the kill; GPU 0/3 are the other user, GPU 2 has an aadivyar orphan).
# Author = Bedrock claude-sonnet-4-6, thinking high (WORK Bedrock cred AWS_BEARER_TOKEN_BEDROCK from the
# project .env, us-east-1 — same work cred as every prior synthesis run; NOT a personal account).
# WIN BAR: beat IterGen 52.0%/94.7% -> >=157/300 accuracy, syntax within ~10-15pp.
set -u
cd /home/aadivyar/csd-generation
export SPIDER_DB_DIR=/home/aadivyar/csd-generation/synthesis/evaluate/syncode/syncode/utils/sql_spider_eval/databases
export CSD_API_MAX_RETRIES=10
OUT=outputs/generated/spider1p5b_spanclose_cold_r2_20260615
mkdir -p "$OUT"
SPIDER_SPLIT=/home/aadivyar/csd-generation/environment/benchmark_splits/spider_dev_proportional_300x300_seed334.json

CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:${LD_LIBRARY_PATH:-} /apps/conda/advayth2/envs/advayth2/bin/python -m synthesis.run_synthesis \
  --task 'Generate a single valid SQL query as exactly `SQL: <<YOUR QUERY>>`, using only the provided schema context.' \
  --dataset spider \
  --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock \
  --eval-model Qwen/Qwen2.5-1.5B-Instruct --eval-backend vllm \
  --max-iterations 20 \
  --output-name spider1p5b_spanclose_cold_r2_20260615 \
  --min-accuracy 0.55 --min-syntax-rate 0.85 \
  --eval-sample-size 300 --eval-max-steps 1200 --eval-step-token-budget 1 \
  --eval-max-seconds-per-example 300 --eval-min-examples-before-threshold-stop 300 \
  --max-tokens 32768 --restart-after-stuck-iters 0 \
  --vllm-gpu-memory-utilization 0.20 --device auto \
  --output-dir "$OUT" \
  --adaptive-helper-mask --helper-selection-policy bandit --refinement-beam-size 2 \
  --anthropic-thinking enabled --anthropic-effort high --anthropic-thinking-display summarized \
  --vllm-tensor-parallel-size 1 \
  --spider-split-file "$SPIDER_SPLIT" --spider-split-name train
echo "SYNTH_EXIT=$?"
echo "DONE_SPIDER1P5B_SPANCLOSE_COLD_R2 $(date)"
