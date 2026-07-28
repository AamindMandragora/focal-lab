#!/usr/bin/env bash
# Emits NEW events from the att2 held-out re-eval (waits through the sweep first).
# Exits on ATT2_ALL_DONE or OOM.
cd /home/aadivyar/csd-generation
DRV=outputs/controlled_comparison/ablation_att2_after_sweep.log
PREV=/tmp/att2_seen.$$
: > "$PREV"
while true; do
  cur=$( {
    grep -hE "ATT2_|REEVAL_DONE|exit=" "$DRV" 2>/dev/null
    f=outputs/generated/ablation_gsm2b_att2_heldout_maxsteps900/run.log
    [ -f "$f" ] && grep -hE "Accuracy:|Syntax:|out of memory|OutOfMemoryError|CUDA out|Traceback" "$f" 2>/dev/null | sed 's#^#[att2_900] #'
  } )
  comm -13 <(sort "$PREV") <(printf '%s\n' "$cur" | sort)
  printf '%s\n' "$cur" > "$PREV"
  if grep -q "ATT2_ALL_DONE" "$DRV" 2>/dev/null; then echo "MONITOR: ATT2_ALL_DONE"; break; fi
  if grep -qiE "out of memory|OutOfMemoryError|CUDA out" outputs/generated/ablation_gsm2b_att2_heldout_maxsteps900/run.log 2>/dev/null; then echo "MONITOR: ATT2 OOM"; break; fi
  sleep 30
done
rm -f "$PREV"
