#!/usr/bin/env bash
# Stream per-attempt CSD rationale + result + hard failures from a synthesis run log,
# tagged by RUN. Feeds the rationale-watch Monitor. Complements watch_cycle1.sh, which
# only greps single-line result/failure signals and does NOT capture the multi-line
# // CSD_RATIONALE_BEGIN..END blocks (that needs the stateful awk capture below).
# Starts at end-of-file (-n 0) so it only emits NEW attempts (avoids re-flooding on
# monitor restart); earlier attempts are seeded into the live log by hand.
# Usage: rationale_emit.sh <TAG> <logfile>
set -u
TAG="${1:?tag}"
LOG="${2:?log}"
tail -n 0 -F "$LOG" 2>/dev/null | awk -v tag="$TAG" '
  /CSD_RATIONALE_BEGIN/ { cap=1; buf=""; next }
  /CSD_RATIONALE_END/   { if (cap) { rat=buf; cap=0 } next }
  cap==1 { buf = buf $0 "\n"; next }
  /Accuracy: [0-9.]+% \(min:/ { acc=$0 }
  /Syntax: [0-9.]+% \(min:/   { syn=$0 }
  /anchor for next refinement: attempt/ {
    print "===== " tag " | " $0
    gsub(/^[ \t]+/, "", acc); gsub(/^[ \t]+/, "", syn);
    if (acc != "") print "  this-attempt " acc
    if (syn != "") print "  this-attempt " syn
    if (rat != "") print rat
    print "===== end " tag " ====="
    acc=""; syn=""; rat=""
    fflush(); next
  }
  /ACCEPTED|Evaluation PASSED|All thresholds met|threshold met|\xe2\x9c\x93 Evaluation/ {
    print "##### " tag " PASS | " $0
    if (rat != "") print rat
    print "##### end " tag " PASS #####"
    fflush(); next
  }
  /Traceback|Error:|FATAL|Killed|OutOfMemory|CUDA out of memory|ThrottlingException|TooManyRequests|\(429\)|HTTP 429|status[^0-9]?429|exit=[1-9]/ {
    print "!!!!! " tag " FAILURE | " $0
    fflush(); next
  }
'
