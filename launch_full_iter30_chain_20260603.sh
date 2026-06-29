#!/bin/bash
# Sequential 4-stage iter-30 chain on GPU 2, 2026-06-03:
#   1) GSM-1.5B-iter30 -> 2) Spider-1.5B-iter30 -> 3) Spider-7B-iter30 -> 4) GSM-7B-iter30
# Each stage waits for the prior python to exit, then sleeps 60s for GPU teardown.
# Archives any pre-existing output dir + prior log before launching that stage.
set -u
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
cd /home/aadivyar/csd-generation
mkdir -p logs/synth_chain
LOG=logs/synth_chain/full_iter30_chain_20260603.log
echo "[full-iter30-chain $(date)] starting" > "$LOG"

# Defensive: wait for any stray synthesis python to exit first.
while pgrep -f "synthesis.run_synthesis" >/dev/null 2>&1; do sleep 60; done
echo "[full-iter30-chain $(date)] no pre-existing synthesis python" >> "$LOG"

run_stage () {
  local stage_name="$1"
  local out_dir="$2"
  local log_file="$3"
  local launcher="$4"
  local match_pat="$5"
  if [ -d "$out_dir" ]; then
    local stamp=$(date +%H%M%S)
    mv "$out_dir" "${out_dir}_archived_${stamp}"
    echo "[full-iter30-chain $(date)] archived prior $stage_name output -> ${out_dir}_archived_${stamp}" >> "$LOG"
  fi
  mv "$log_file" "${log_file}.prev" 2>/dev/null || true
  echo "[full-iter30-chain $(date)] launching $stage_name" >> "$LOG"
  bash "$launcher" >> "$LOG" 2>&1
  sleep 10
  while pgrep -f "synthesis.run_synthesis .*${match_pat}" >/dev/null 2>&1; do
    sleep 60
  done
  echo "[full-iter30-chain $(date)] $stage_name python exited, sleeping 60s for GPU teardown" >> "$LOG"
  sleep 60
}

run_stage "GSM-1.5B-iter30" \
  "outputs/generated/ralph_1p5B_gsm_disjoint_iter30_20260603" \
  "/tmp/ralph_1p5B_gsm_disjoint_iter30_20260603.log" \
  "launch_gsm1p5b_disjoint_iter30_20260603.sh" \
  "ralph_1p5B_gsm_disjoint_iter30_20260603"

run_stage "Spider-1.5B-iter30" \
  "outputs/generated/ralph_1p5B_spider_disjoint_iter30_20260603" \
  "/tmp/ralph_1p5B_spider_disjoint_iter30_20260603.log" \
  "launch_spider1p5b_disjoint_iter30_20260603.sh" \
  "ralph_1p5B_spider_disjoint_iter30_20260603"

run_stage "Spider-7B-iter30" \
  "outputs/generated/ralph_7B_spider_disjoint_iter30_20260603" \
  "/tmp/ralph_7B_spider_disjoint_iter30_20260603.log" \
  "launch_spider7b_disjoint_iter30_20260603.sh" \
  "ralph_7B_spider_disjoint_iter30_20260603"

run_stage "GSM-7B-iter30" \
  "outputs/generated/ralph_7B_gsm_disjoint_iter30_20260603" \
  "/tmp/ralph_7B_gsm_disjoint_iter30_20260603.log" \
  "launch_gsm7b_disjoint_iter30_20260603.sh" \
  "ralph_7B_gsm_disjoint_iter30_20260603"

echo "[full-iter30-chain $(date)] all 4 stages complete" >> "$LOG"
