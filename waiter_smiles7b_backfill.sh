#!/bin/bash
# After the first 7B SMILES lane pass ends, relaunch the lane once to backfill:
# - acrylates synthesis (first pass was wasted by the RDKit hard-gate bug)
# - any itergen/gcd baselines that OOM'd while stale procs held GPU 1
# The patched smiles_lane.sh skips classes that already have metadecode.json.
set -uo pipefail
cd /home/aadivyar/csd-generation
LOG=outputs/smiles_lane_7B.log

until grep -q DONE_SMILES_LANE_7B "$LOG" 2>/dev/null; do sleep 120; done
echo "=== BACKFILL PASS smiles_7B START $(date) ===" >> "$LOG"
# GPU 2, not 1: the stale procs squatting GPU 1 OOM'd itergen/gcd on the first pass,
# and GPU 2 is fully free once the GSM-7B chain (synthesis + re-eval) is done.
bash smiles_lane.sh Qwen/Qwen2.5-7B-Instruct 7B 2 0.45 >> "$LOG" 2>&1
echo "BACKFILL_7B_EXIT=$?"
echo DONE_SMILES_BACKFILL_7B
