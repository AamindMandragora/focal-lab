#!/bin/bash
# One-line status summary for the Change-3 probe. Printed each Monitor poll.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"
L=outputs/generated/spider1p5b_change3_probe_20260615_run.log
att=$(grep -cE "Attempt [0-9]+/3" "$L" 2>/dev/null)
verify=$(grep -E "[0-9]+ verified, [0-9]+ error|VERIFICATION (PASSED|FAILED|OK)|verification (passed|failed)" "$L" 2>/dev/null | tail -1 | tr -s ' ' | cut -c1-55)
mspan=$(grep -c "GenerateWithManagedSpan" "$L" 2>/dev/null)
ground=$(grep -c "RegenerateUnitOnGroundingFailure" "$L" 2>/dev/null)
pen=$(grep -c "\[recurrence\] penalize" "$L" 2>/dev/null)
acc=$(grep -iE "accuracy[^a-z]|syntax[_ ]?rate|eval result|examples ran" "$L" 2>/dev/null | tail -1 | tr -s ' ' | cut -c1-70)
err=$(grep -E "Traceback|Error:|FAILED|0000000000|CXXABI|RuntimeError|TimeoutError" "$L" 2>/dev/null | tail -1 | tr -s ' ' | cut -c1-70)
done=$(grep -c "DONE_SPIDER1P5B_CHANGE3_PROBE" "$L" 2>/dev/null)
alive=$(pgrep -u aadivyar -f synthesis.run_synthesis >/dev/null && echo 1 || echo 0)
state="RUN"
[ "$done" -ge 1 ] 2>/dev/null && state="DONE"
{ [ "$alive" = "0" ] && [ "$done" -lt 1 ]; } 2>/dev/null && state="DEAD"
echo "[$state] att=$att/3 mspan=$mspan ground=$ground penalize=$pen | verify:${verify:-?} | eval:${acc:-none} | err:${err:-none}"
