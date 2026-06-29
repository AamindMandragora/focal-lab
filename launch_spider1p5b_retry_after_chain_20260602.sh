#!/bin/bash
# Wait for the disjoint-split chain wrapper (pid 3281868) to exit, then
# re-run Spider-1.5B with --max-iterations 20 (now baked into its launcher).
# Detach with setsid nohup so this script outlives any ssh handle.
set -u
cd /home/aadivyar/csd-generation
mkdir -p logs/synth_chain
LOG=logs/synth_chain/spider1p5b_retry_20260602.log
echo "[retry $(date)] waiting for chain wrapper 3281868 to exit" > "$LOG"
while ps -p 3281868 >/dev/null 2>&1; do sleep 60; done
echo "[retry $(date)] chain wrapper exited, sleeping 60s for GPU teardown" >> "$LOG"
sleep 60
# Move first-run output dir aside so the synthesis system starts clean.
OLD=outputs/generated/ralph_1p5B_spider_disjoint_20260602
if [ -d "$OLD" ]; then
  STAMP=$(date +%H%M%S)
  mv "$OLD" "${OLD}_attempt1_archived_${STAMP}"
  echo "[retry $(date)] archived prior output dir -> ${OLD}_attempt1_archived_${STAMP}" >> "$LOG"
fi
mv /tmp/ralph_1p5B_spider_disjoint_20260602.log "/tmp/ralph_1p5B_spider_disjoint_20260602.attempt1.log" 2>/dev/null || true
echo "[retry $(date)] launching spider-1.5B with max-iterations=20" >> "$LOG"
bash launch_spider1p5b_disjoint_20260602.sh >> "$LOG" 2>&1
sleep 10
while pgrep -f "synthesis.run_synthesis .*ralph_1p5B_spider_disjoint_20260602" >/dev/null 2>&1; do
  sleep 60
done
echo "[retry $(date)] spider-1.5B retry python exited" >> "$LOG"
