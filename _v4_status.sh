#!/bin/bash
# One status line for the v4 flat-penalty re-eval; exit 42 when DONE marker present.
LOG=/home/aadivyar/csd-generation/outputs/generated/spider1p5b_flatpenalty_v4_reeval_20260616/run.log
if [ ! -f "$LOG" ]; then
  echo "[v4] log not created yet"
  exit 0
fi
mode=$(grep -oE "penalty mode=[a-z(),]+" "$LOG" 2>/dev/null | head -1)
grounded=$(grep -c "unit fully grounded" "$LOG" 2>/dev/null)
located=$(grep -c "first-ungrounded token_idx" "$LOG" 2>/dev/null)
penalties=$(grep -c "at prefix_len=" "$LOG" 2>/dev/null)
maxcount=$(grep -oE "counts now=\{[0-9]+: [0-9]+\}" "$LOG" 2>/dev/null | grep -oE "[0-9]+\}" | tr -d '}' | sort -n | tail -1)
errors=$(grep -cE "Traceback|Error:|FAILED|AssertionError|Killed|out of memory|RuntimeError|CUDA error" "$LOG" 2>/dev/null)
exit_line=$(grep "REEVAL_EXIT=" "$LOG" 2>/dev/null | tail -1)
done=$(grep -c "DONE_GROUNDING_REEVAL_V4" "$LOG" 2>/dev/null)
last=$(grep -vE "^\s*$" "$LOG" 2>/dev/null | tail -1 | cut -c1-150)
echo "[v4] $mode grounded=$grounded located=$located penalties=$penalties maxcount=$maxcount errors=$errors done=$done ${exit_line} | tail: $last"
if [ "$done" -gt 0 ]; then
  exit 42
fi
exit 0
