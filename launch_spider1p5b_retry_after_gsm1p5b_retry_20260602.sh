#!/bin/bash
# Re-queue Spider-1.5B retry after the in-flight GSM-1.5B retry exits.
# The first Spider-1.5B retry wrapper (pid 279583) crashed at launch because
# launch_spider1p5b_disjoint_20260602.sh referenced $LD_LIBRARY_PATH under
# set -u with the var unset in the retry wrapper's env. Launcher now uses
# ${LD_LIBRARY_PATH:-}; this wrapper kicks off the actual run once GPU 2
# frees up.
# Wait condition: GSM-1.5B retry python pid 2156177 (active synthesis run).
# Detached via setsid nohup at invocation time.
# Pre-export LD_LIBRARY_PATH defensively so even older un-patched launchers
# would survive if invoked here by mistake.
set -u
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
cd /home/aadivyar/csd-generation
mkdir -p logs/synth_chain
LOG=logs/synth_chain/spider1p5b_retry_b_20260602.log
echo "[retry-b $(date)] waiting for gsm-1.5B retry python (pid 2156177) to exit" > "$LOG"
while ps -p 2156177 >/dev/null 2>&1; do sleep 60; done
echo "[retry-b $(date)] gsm-1.5B retry python exited, sleeping 60s for GPU teardown" >> "$LOG"
sleep 60
OLD=outputs/generated/ralph_1p5B_spider_disjoint_20260602
if [ -d "$OLD" ]; then
  STAMP=$(date +%H%M%S)
  mv "$OLD" "${OLD}_attempt2_archived_${STAMP}"
  echo "[retry-b $(date)] archived prior empty spider output dir -> ${OLD}_attempt2_archived_${STAMP}" >> "$LOG"
fi
mv /tmp/ralph_1p5B_spider_disjoint_20260602.log "/tmp/ralph_1p5B_spider_disjoint_20260602.attempt2.log" 2>/dev/null || true
echo "[retry-b $(date)] launching spider-1.5B with patched launcher (max-iterations=20, LD safe)" >> "$LOG"
bash launch_spider1p5b_disjoint_20260602.sh >> "$LOG" 2>&1
sleep 10
while pgrep -f "synthesis.run_synthesis .*ralph_1p5B_spider_disjoint_20260602" >/dev/null 2>&1; do
  sleep 60
done
echo "[retry-b $(date)] spider-1.5B retry-b python exited" >> "$LOG"
