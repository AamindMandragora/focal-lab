#!/bin/bash
# One status line for the span-close-feedback COLD Spider-1.5B run. Tracks per-attempt scores,
# verify-fails, and the NEW span-close signal: how many outputs opened-but-never-closed, and the
# reported preamble-share % (does it drop = author breaking the preamble-heavy shape?).
# exit 42 when DONE marker present.
LOG=/home/aadivyar/csd-generation/outputs/generated/spider1p5b_spanclose_cold_20260616/run.log
if [ ! -f "$LOG" ]; then
  echo "[spanclose] log not created yet"
  exit 0
fi
attempt=$(grep -oE "Attempt [0-9]+/[0-9]+" "$LOG" 2>/dev/null | tail -1 | tr -d ' ')
evalex=$(grep -oE "Processing example [0-9]+/[0-9]+" "$LOG" 2>/dev/null | tail -1 | grep -oE "[0-9]+/[0-9]+")
verifyfail=$(grep -c "Verification failed" "$LOG" 2>/dev/null)
attstarted=$(grep -cE "^Attempt [0-9]+/[0-9]+" "$LOG" 2>/dev/null)
# span-close signal: latest "Span-closure check: N/M" and latest reported preamble share %
spanclose=$(grep -oE "Span-closure check: [0-9]+/[0-9]+" "$LOG" 2>/dev/null | tail -1 | grep -oE "[0-9]+/[0-9]+")
preamble=$(grep -oE "did not open until on average [0-9]+%" "$LOG" 2>/dev/null | tail -1 | grep -oE "[0-9]+%")
acc=$(grep -oiE "accuracy: [0-9.]+%" "$LOG" 2>/dev/null | tail -1 | grep -oE "[0-9.]+")
syn=$(grep -oiE "syntax: [0-9.]+%" "$LOG" 2>/dev/null | tail -1 | grep -oE "[0-9.]+")
accepted=$(grep -cE "NEW BEST|ACCEPTED|new best strategy|Accepted new" "$LOG" 2>/dev/null)
errors=$(grep -cE "Traceback|FATAL|Killed|out of memory|CUDA error|URLError|quota|MemoryError" "$LOG" 2>/dev/null)
exit_line=$(grep "SYNTH_EXIT=" "$LOG" 2>/dev/null | tail -1)
done=$(grep -c "DONE_SPIDER1P5B_SPANCLOSE_COLD" "$LOG" 2>/dev/null)
last=$(grep -vE "^\s*$" "$LOG" 2>/dev/null | tail -1 | cut -c1-130)
echo "[spanclose] ${attempt:-Attempt?} eval_ex=${evalex:-none} verifyFAIL=$verifyfail/$attstarted spanNotClosed=${spanclose:-NA} preamble%=${preamble:-NA} last_acc=${acc:-NA} last_syn=${syn:-NA} accepts=$accepted err=$errors done=$done ${exit_line} | tail: $last"
if [ "$done" -gt 0 ]; then
  exit 42
fi
exit 0
