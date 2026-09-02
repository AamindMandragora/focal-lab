#!/bin/bash
# Sequential 7B iter-30 retry chain (Spider then GSM) that waits for the
# entire in-flight 1.5B retry chain (gsm-1.5B retry + spider-1.5B retry-b)
# to fully drain on GPU 2 before launching. Detached.
# Wait condition: no remaining `synthesis.run_synthesis` python AND no
# `spider1p5b_retry_after_gsm1p5b_retry` wrapper. Then 60s GPU teardown.
# After Spider-7B-iter30 python exits, sleep 60s and start GSM-7B-iter30.
set -u
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
cd /home/aadivyar/csd-generation
mkdir -p logs/synth_chain
LOG=logs/synth_chain/7b_iter30_chain_20260603.log
echo "[7b-iter30-chain $(date)] waiting for any active synthesis python to exit" > "$LOG"
while pgrep -f "synthesis.run_synthesis" >/dev/null 2>&1; do sleep 60; done
echo "[7b-iter30-chain $(date)] no active synthesis python" >> "$LOG"
echo "[7b-iter30-chain $(date)] waiting for spider-1.5B retry-b wrapper to exit (if still alive)" >> "$LOG"
while pgrep -f "spider1p5b_retry_after_gsm1p5b_retry" >/dev/null 2>&1; do sleep 60; done
echo "[7b-iter30-chain $(date)] all 1.5B chain wrappers gone, sleeping 60s for GPU teardown" >> "$LOG"
sleep 60

# Stage 1: Spider-7B iter30
OLD1=outputs/generated/ralph_7B_spider_disjoint_iter30_20260603
if [ -d "$OLD1" ]; then
  STAMP=$(date +%H%M%S)
  mv "$OLD1" "${OLD1}_archived_${STAMP}"
  echo "[7b-iter30-chain $(date)] archived prior spider-7B-iter30 output -> ${OLD1}_archived_${STAMP}" >> "$LOG"
fi
mv /tmp/ralph_7B_spider_disjoint_iter30_20260603.log "/tmp/ralph_7B_spider_disjoint_iter30_20260603.prev.log" 2>/dev/null || true
echo "[7b-iter30-chain $(date)] launching Spider-7B iter30" >> "$LOG"
bash launch_spider7b_disjoint_iter30_20260603.sh >> "$LOG" 2>&1
sleep 10
while pgrep -f "synthesis.run_synthesis .*ralph_7B_spider_disjoint_iter30_20260603" >/dev/null 2>&1; do
  sleep 60
done
echo "[7b-iter30-chain $(date)] Spider-7B iter30 python exited, sleeping 60s for GPU teardown" >> "$LOG"
sleep 60

# Stage 2: GSM-7B iter30
OLD2=outputs/generated/ralph_7B_gsm_disjoint_iter30_20260603
if [ -d "$OLD2" ]; then
  STAMP=$(date +%H%M%S)
  mv "$OLD2" "${OLD2}_archived_${STAMP}"
  echo "[7b-iter30-chain $(date)] archived prior gsm-7B-iter30 output -> ${OLD2}_archived_${STAMP}" >> "$LOG"
fi
mv /tmp/ralph_7B_gsm_disjoint_iter30_20260603.log "/tmp/ralph_7B_gsm_disjoint_iter30_20260603.prev.log" 2>/dev/null || true
echo "[7b-iter30-chain $(date)] launching GSM-7B iter30" >> "$LOG"
bash launch_gsm7b_disjoint_iter30_20260603.sh >> "$LOG" 2>&1
sleep 10
while pgrep -f "synthesis.run_synthesis .*ralph_7B_gsm_disjoint_iter30_20260603" >/dev/null 2>&1; do
  sleep 60
done
echo "[7b-iter30-chain $(date)] GSM-7B iter30 python exited; 7B iter30 chain complete" >> "$LOG"
