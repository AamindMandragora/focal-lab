#!/bin/bash
# The running 1.5B lane (PID 2592678) parsed smiles_lane.sh BEFORE later edits; when its
# loop ends, bash re-reads the edited file at stale byte offsets and dies WITHOUT printing
# the DONE marker (this killed the 7B lane's tail). Watch the process and guarantee the
# marker so downstream waiters (acrylates retry, backfill) can fire.
set -uo pipefail
cd /home/aadivyar/csd-generation
LOG=outputs/smiles_lane_1p5B.log
PID=2592678

while kill -0 "$PID" 2>/dev/null; do sleep 60; done
sleep 5
if ! grep -q DONE_SMILES_LANE_1P5B "$LOG" 2>/dev/null; then
  echo "DONE_SMILES_LANE_1P5B (appended by supervisor; lane died at post-loop script corruption)" >> "$LOG"
fi
echo SUPERVISOR_1P5B_DONE
