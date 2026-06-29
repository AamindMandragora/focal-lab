#!/bin/bash
# After the first 1.5B SMILES lane pass ends, relaunch the lane once with the FIXED
# on-disk script (success_report-based CSD pickup, accepted-strategy reuse, metadecode
# skip). The first pass ran old in-memory code: its chain_extenders/isocyanates synthesis
# results get picked up properly here, the OOM'd gcd-acrylates baseline retries, and any
# missing synthesis re-runs. Runs AFTER the acrylates retry waiter check (that waiter
# exits early if metadecode.json exists, and this lane skips synthesis for classes the
# retry already completed).
set -uo pipefail
cd /home/aadivyar/csd-generation
LOG=outputs/smiles_lane_1p5B.log

until grep -q DONE_SMILES_LANE_1P5B "$LOG" 2>/dev/null; do sleep 120; done
# Serialize behind the acrylates warmstart retry (same GPU, same class): wait until that
# waiter either finished its chain or skipped.
until grep -qE "DONE_ACRYLATES_1P5B_RETRY|SKIP retry" outputs/waiter_smiles1p5b_acrylates_retry.log 2>/dev/null; do sleep 120; done
echo "=== BACKFILL PASS smiles_1p5B START $(date) ===" >> "$LOG"
bash smiles_lane.sh Qwen/Qwen2.5-1.5B-Instruct 1p5B 0 0.30 >> "$LOG" 2>&1
echo "BACKFILL_1P5B_EXIT=$?"
echo DONE_SMILES_BACKFILL_1P5B
