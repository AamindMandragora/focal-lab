#!/bin/bash
# Wait for the Spider-1.5B retry wrapper (will set its python pid) to exit, then
# re-run GSM-1.5B with --max-iterations 20 (baked into launcher) on the
# threshold-impossible-removed evaluator. Detached.
set -u
cd /home/aadivyar/csd-generation
mkdir -p logs/synth_chain
LOG=logs/synth_chain/gsm1p5b_retry_20260602.log
echo "[gsm-retry $(date)] waiting for spider-1.5B retry wrapper to exit" > "$LOG"
# Wait for the spider-1.5B retry wrapper to disappear
while pgrep -f "spider1p5b_retry_after_chain" >/dev/null 2>&1; do sleep 60; done
echo "[gsm-retry $(date)] spider-1.5B retry wrapper gone, sleeping 60s for GPU teardown" >> "$LOG"
sleep 60
OLD=outputs/generated/ralph_1p5B_gsm_disjoint_20260602
if [ -d "$OLD" ]; then
  STAMP=$(date +%H%M%S)
  mv "$OLD" "${OLD}_attempt1_archived_${STAMP}"
  echo "[gsm-retry $(date)] archived prior output dir -> ${OLD}_attempt1_archived_${STAMP}" >> "$LOG"
fi
mv /tmp/ralph_1p5B_gsm_disjoint_20260602.log "/tmp/ralph_1p5B_gsm_disjoint_20260602.attempt1.log" 2>/dev/null || true
echo "[gsm-retry $(date)] launching gsm-1.5B with max-iterations=20 + patched evaluator" >> "$LOG"
bash launch_gsm1p5b_disjoint_20260602.sh >> "$LOG" 2>&1
sleep 10
while pgrep -f "synthesis.run_synthesis .*ralph_1p5B_gsm_disjoint_20260602" >/dev/null 2>&1; do
  sleep 60
done
echo "[gsm-retry $(date)] gsm-1.5B retry python exited" >> "$LOG"
