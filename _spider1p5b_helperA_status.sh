#!/bin/bash
# One status line for the verify-friction-helper COLD Spider-1.5B run. PRIMARY signal: verify-fail
# rate (target well below the last run's 4/7) and whether the author USES the new abstraction
# (GenerateWithPrefixAndManagedSpan call count). exit 42 when DONE marker present.
LOG=/home/aadivyar/csd-generation/outputs/generated/spider1p5b_helperA_cold_20260617/run.log
if [ ! -f "$LOG" ]; then
  echo "[helperA] log not created yet"
  exit 0
fi
attempt=$(grep -oE "Attempt [0-9]+/[0-9]+" "$LOG" 2>/dev/null | tail -1 | tr -d ' ')
evalex=$(grep -oE "Processing example [0-9]+/[0-9]+" "$LOG" 2>/dev/null | tail -1 | grep -oE "[0-9]+/[0-9]+")
verifyfail=$(grep -c "Verification failed" "$LOG" 2>/dev/null)
attstarted=$(grep -cE "^Attempt [0-9]+/[0-9]+" "$LOG" 2>/dev/null)
# does the author actually USE the new abstraction?
useshelperA=$(grep -cE "helpers\.GenerateWithPrefixAndManagedSpan\(" "$LOG" 2>/dev/null)
acc=$(grep -oiE "accuracy: [0-9.]+%" "$LOG" 2>/dev/null | tail -1 | grep -oE "[0-9.]+")
syn=$(grep -oiE "syntax: [0-9.]+%" "$LOG" 2>/dev/null | tail -1 | grep -oE "[0-9.]+")
accepted=$(grep -cE "NEW BEST|ACCEPTED|new best strategy|Accepted new" "$LOG" 2>/dev/null)
errors=$(grep -cE "Traceback|FATAL|Killed|out of memory|CUDA error|URLError|quota|MemoryError" "$LOG" 2>/dev/null)
exit_line=$(grep "SYNTH_EXIT=" "$LOG" 2>/dev/null | tail -1)
done=$(grep -c "DONE_SPIDER1P5B_HELPERA_COLD" "$LOG" 2>/dev/null)
last=$(grep -vE "^\s*$" "$LOG" 2>/dev/null | tail -1 | cut -c1-130)
echo "[helperA] ${attempt:-Attempt?} eval_ex=${evalex:-none} verifyFAIL=$verifyfail/$attstarted usesHelperA=$useshelperA last_acc=${acc:-NA} last_syn=${syn:-NA} accepts=$accepted err=$errors done=$done ${exit_line} | tail: $last"
if [ "$done" -gt 0 ]; then
  exit 42
fi
exit 0
