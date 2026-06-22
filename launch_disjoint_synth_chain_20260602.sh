#!/bin/bash
# Chain the 4 disjoint-split synth runs on GPU 2 sequentially, smallest first.
# Each step launches its own script (which backgrounds the python process), then
# this wrapper waits until that python process exits before starting the next.
# Single chain log at logs/synth_chain/disjoint_chain_20260602.log so progress
# is tailable in one place.
set -u
cd /home/aadivyar/csd-generation
mkdir -p logs/synth_chain
CHAIN_LOG=logs/synth_chain/disjoint_chain_20260602.log
echo "[chain $(date)] starting disjoint-split synth chain on GPU 2" > "$CHAIN_LOG"

run_step() {
  local label="$1"; local launcher="$2"; local pattern="$3"
  echo "[chain $(date)] starting $label via $launcher" >> "$CHAIN_LOG"
  bash "$launcher" >> "$CHAIN_LOG" 2>&1
  sleep 10
  # wait until the python process for this run exits
  while pgrep -f "$pattern" >/dev/null 2>&1; do
    sleep 60
  done
  echo "[chain $(date)] $label python process exited" >> "$CHAIN_LOG"
  sleep 30  # vLLM teardown breathing room
}

run_step "spider-1.5B" launch_spider1p5b_disjoint_20260602.sh \
  "synthesis.run_synthesis .*ralph_1p5B_spider_disjoint_20260602"
run_step "gsm-1.5B"    launch_gsm1p5b_disjoint_20260602.sh \
  "synthesis.run_synthesis .*ralph_1p5B_gsm_disjoint_20260602"
run_step "spider-7B"   launch_spider7b_disjoint_20260602.sh \
  "synthesis.run_synthesis .*ralph_7B_spider_disjoint_20260602"
run_step "gsm-7B"      launch_gsm7b_disjoint_20260602.sh \
  "synthesis.run_synthesis .*ralph_7B_gsm_disjoint_20260602"

echo "[chain $(date)] disjoint-split synth chain complete" >> "$CHAIN_LOG"
