#!/bin/bash
# One status line for the full COLD Spider-1.5B symbol-recover run. exit 42 when DONE marker present.
LOG=/home/aadivyar/csd-generation/outputs/generated/spider1p5b_symbolrecover_cold_20260616/run.log
if [ ! -f "$LOG" ]; then
  echo "[symbolrecover] log not created yet"
  exit 0
fi
attempt=$(grep -oE "Attempt [0-9]+/[0-9]+" "$LOG" 2>/dev/null | tail -1 | tr -d ' ')
evalex=$(grep -oE "Processing example [0-9]+/[0-9]+" "$LOG" 2>/dev/null | tail -1 | grep -oE "[0-9]+/[0-9]+")
acc=$(grep -oiE "accuracy[^0-9]*[0-9.]+" "$LOG" 2>/dev/null | tail -1 | grep -oE "[0-9.]+")
syn=$(grep -oiE "syntax[^0-9]*[0-9.]+" "$LOG" 2>/dev/null | tail -1 | grep -oE "[0-9.]+")
accepted=$(grep -cE "NEW BEST|ACCEPTED|new best strategy|Accepted new" "$LOG" 2>/dev/null)
grounded=$(grep -c "unit fully grounded" "$LOG" 2>/dev/null)
errors=$(grep -cE "Traceback|FATAL|Killed|out of memory|CUDA error|URLError|quota|MemoryError" "$LOG" 2>/dev/null)
exit_line=$(grep "SYNTH_EXIT=" "$LOG" 2>/dev/null | tail -1)
done=$(grep -c "DONE_SPIDER1P5B_SYMBOLRECOVER_COLD" "$LOG" 2>/dev/null)
last=$(grep -vE "^\s*$" "$LOG" 2>/dev/null | tail -1 | cut -c1-160)
echo "[symbolrecover] ${attempt:-Attempt?} eval_ex=${evalex:-none} last_acc=${acc:-NA} last_syn=${syn:-NA} accepts=$accepted grounded_units=$grounded errlines=$errors done=$done ${exit_line} | tail: $last"
if [ "$done" -gt 0 ]; then
  exit 42
fi
exit 0
