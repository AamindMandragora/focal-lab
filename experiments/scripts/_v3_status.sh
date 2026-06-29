#!/bin/bash
# Prints ONE status line for the v3 symbol-boundary re-eval, then exits.
# Exit 42 signals completion (DONE marker present) so the Monitor loop can stop.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"
LOG=$ROOT/outputs/generated/spider1p5b_symbolboundary_v3_reeval_20260616/run.log
if [ ! -f "$LOG" ]; then
  echo "[v3] log not created yet"
  exit 0
fi
grounded=$(grep -c "unit fully grounded" "$LOG" 2>/dev/null)
located=$(grep -c "first-ungrounded token_idx" "$LOG" 2>/dev/null)
penalties=$(grep -c "at prefix_len=" "$LOG" 2>/dev/null)
errors=$(grep -cE "Traceback|Error:|FAILED|AssertionError|Killed|out of memory|RuntimeError|CUDA error" "$LOG" 2>/dev/null)
verified=$(grep -c "verified, 0 errors" "$LOG" 2>/dev/null)
exit_line=$(grep "REEVAL_EXIT=" "$LOG" 2>/dev/null | tail -1)
done=$(grep -c "DONE_GROUNDING_REEVAL_V3" "$LOG" 2>/dev/null)
last=$(grep -vE "^\s*$" "$LOG" 2>/dev/null | tail -1 | cut -c1-160)
echo "[v3] grounded=$grounded located=$located penalties=$penalties verified=$verified errors=$errors done=$done ${exit_line} | tail: $last"
if [ "$done" -gt 0 ]; then
  exit 42
fi
exit 0
