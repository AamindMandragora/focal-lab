#!/usr/bin/env bash
# Tail the newest cycle-1 cold log matching the glob in $1 and emit only real signals:
# attempt boundaries, accuracy/syntax scores, accept/reject verdicts, completion, crash, 429.
# Excludes per-attempt profiling dumps (calls=/avg=/.total/_mask. etc). Used by Monitor.
set -u
GLOB="${1:?usage: watch_cycle1.sh <log-glob> [scan]}"
MODE="${2:-follow}"   # follow = tail -f (fragile over SSH); scan = grep once and exit (poll-friendly)
L=$(ls -t $GLOB 2>/dev/null | head -1)
[ -z "$L" ] && { echo "NO_LOG_YET for $GLOB"; exit 1; }
if [ "$MODE" = "scan" ]; then
  grep -anE "Attempt [0-9]+/[0-9]+|Final accuracy|Accuracy: [0-9]|Syntax rate:|Strategy (accepted|rejected)|ACCEPTED|REJECTED|NEW BEST|ALL DONE|exit=|Too [Mm]any [Rr]equest|TooManyRequest|ThrottlingException|\(429\)|status.{0,5}429|[Qq]uota exceed|Throttl|[Rr]ate.?limit|Traceback .most recent|Killed|OutOfMemory|CUDA out of memory|died unexpectedly" "$L" \
    | grep -vE "calls=|avg=|\.total|_mask\.|GenerateLogits|inc_parser|compute_dfa|cache_hit|to_cpu|lookup_next|STEP_BREAKDOWN|end_of_example|callbacks=|wall=|EngineCore"
  exit 0
fi
tail -f "$L" \
  | grep -E --line-buffered "Attempt [0-9]+/[0-9]+|Final accuracy|Accuracy: [0-9]|Syntax rate:|Strategy (accepted|rejected)|ACCEPTED|REJECTED|NEW BEST|ALL DONE|exit=|429|[Qq]uota exceed|Throttl|[Rr]ate.?limit|Traceback .most recent|Killed|OutOfMemory|CUDA out of memory|died unexpectedly|EngineCore" \
  | grep -vE "calls=|avg=|\.total|_mask\.|GenerateLogits|inc_parser|compute_dfa|cache_hit|to_cpu|lookup_next"
