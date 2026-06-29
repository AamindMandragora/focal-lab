#!/bin/bash
# One status line for the friction-fix COLD Spider-1.5B run. Tracks the verify-fail rate (the metric
# the fix targets). exit 42 when DONE marker present.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"
LOG=$ROOT/outputs/generated/spider1p5b_frictionfix_cold_20260616/run.log
if [ ! -f "$LOG" ]; then
  echo "[frictionfix] log not created yet"
  exit 0
fi
attempt=$(grep -oE "Attempt [0-9]+/[0-9]+" "$LOG" 2>/dev/null | tail -1 | tr -d ' ')
evalex=$(grep -oE "Processing example [0-9]+/[0-9]+" "$LOG" 2>/dev/null | tail -1 | grep -oE "[0-9]+/[0-9]+")
verifyfail=$(grep -c "Verification failed" "$LOG" 2>/dev/null)
belowbar=$(grep -c "Evaluation below threshold" "$LOG" 2>/dev/null)
# count attempts started so far (denominator for verify-fail rate)
attstarted=$(grep -cE "^Attempt [0-9]+/[0-9]+" "$LOG" 2>/dev/null)
# scaffold uptake: did the author call the now-visible budget-carrying scaffolds?
rollout=$(grep -cE "helpers\.RolloutConstrainedWithPenalties\(" "$LOG" 2>/dev/null)
congen=$(grep -cE "helpers\.ConstrainedGeneration\(" "$LOG" 2>/dev/null)
managed=$(grep -cE "helpers\.GenerateWithManagedSpan\(" "$LOG" 2>/dev/null)
acc=$(grep -oiE "accuracy[^0-9]*[0-9.]+%" "$LOG" 2>/dev/null | tail -1 | grep -oE "[0-9.]+")
syn=$(grep -oiE "syntax[^0-9]*[0-9.]+%" "$LOG" 2>/dev/null | tail -1 | grep -oE "[0-9.]+")
accepted=$(grep -cE "NEW BEST|ACCEPTED|new best strategy|Accepted new" "$LOG" 2>/dev/null)
errors=$(grep -cE "Traceback|FATAL|Killed|out of memory|CUDA error|URLError|quota|MemoryError" "$LOG" 2>/dev/null)
exit_line=$(grep "SYNTH_EXIT=" "$LOG" 2>/dev/null | tail -1)
done=$(grep -c "DONE_SPIDER1P5B_FRICTIONFIX_COLD" "$LOG" 2>/dev/null)
last=$(grep -vE "^\s*$" "$LOG" 2>/dev/null | tail -1 | cut -c1-150)
echo "[frictionfix] ${attempt:-Attempt?} eval_ex=${evalex:-none} verifyFAIL=$verifyfail/$attstarted scaffolds{rollout=$rollout congen=$congen managed=$managed} last_acc=${acc:-NA} last_syn=${syn:-NA} accepts=$accepted err=$errors done=$done ${exit_line} | tail: $last"
if [ "$done" -gt 0 ]; then
  exit 42
fi
exit 0
