#!/bin/bash
# Emit each NEW synthesis-iteration marker line as it appears, then exit when the
# run process dies (covers both normal finish and crash). Gives one notification
# per iteration boundary/result instead of only at the very end.
# Usage: watch_iterations.sh <logpath> <pid>
LOG="$1"; PID="$2"
seen=0
# Iteration boundaries + per-attempt verdicts + terminal/crash signatures.
# Deliberately EXCLUDES per-example "Processing example k/50" noise.
PAT='Attempt [0-9]+/[0-9]+|Verification passed|Verification FAILED|[Vv]erification failed|Evaluation below threshold|Evaluation passed|Accept(ed|ing) strategy|Accuracy: [0-9]|Syntax: [0-9]|syntax_rate|Early stopping synthesis|best possible accuracy|[Nn]ew best|Best so far|Refining based on|SUCCESS|All targets|Traceback|Error:|CUDA out of memory|OutOfMemory|Killed|RuntimeError|AssertionError'
while true; do
  mapfile -t M < <(grep -aE "$PAT" "$LOG" 2>/dev/null)
  n=${#M[@]}
  if [ "$n" -gt "$seen" ]; then
    for ((i=seen; i<n; i++)); do echo "${M[$i]}"; done
    seen=$n
  fi
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "=== PID $PID EXITED at $(date) -- run finished or crashed; check results report ==="
    break
  fi
  sleep 45
done
