#!/bin/bash
# One-line status for the grounding re-eval. Includes a BILLING GUARD: if any author
# (bedrock/anthropic) API call appears, surface it loudly so we can kill immediately.
cd /home/aadivyar/csd-generation
L=outputs/generated/spider1p5b_groundfix_reeval_20260615_run.log
gfail=$(grep -cE "grounded=False" "$L" 2>/dev/null)
gall=$(grep -cE "\[grounding\] span=" "$L" 2>/dev/null)
pen=$(grep -cE "\[recurrence\] penalize" "$L" 2>/dev/null)
acc=$(grep -iE "Accuracy:|Syntax Rate:|examples\)" "$L" 2>/dev/null | tail -2 | tr '\n' ' ' | tr -s ' ' | cut -c1-80)
bill=$(grep -ciE "bedrock|anthropic|invoke_model|messages.create|InvokeModel" "$L" 2>/dev/null)
err=$(grep -E "Traceback|Error:|FAILED|CXXABI|RuntimeError|TimeoutError" "$L" 2>/dev/null | tail -1 | tr -s ' ' | cut -c1-60)
done=$(grep -c "DONE_GROUNDING_REEVAL" "$L" 2>/dev/null)
alive=$(pgrep -u aadivyar -f synthesis.run_synthesis >/dev/null && echo 1 || echo 0)
state="RUN"; [ "$done" -ge 1 ] 2>/dev/null && state="DONE"
{ [ "$alive" = "0" ] && [ "$done" -lt 1 ]; } 2>/dev/null && state="DEAD"
guard="ok"; [ "$bill" -ge 1 ] 2>/dev/null && guard="!!AUTHOR-CALL($bill)!!"
echo "[$state] grounding_fail=$gfail/$gall penalize=$pen billing=$guard | eval:${acc:-none} | err:${err:-none}"
