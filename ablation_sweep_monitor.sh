#!/usr/bin/env bash
# Emits one line per NEW event from the GSM-2B max-steps sweep: step markers, per-step
# held-out Accuracy/Syntax, and any OOM/crash signature. Exits on SWEEP_ALL_DONE or OOM.
cd /home/aadivyar/csd-generation
SWEEP=outputs/controlled_comparison/ablation_gsm2b_maxsteps_sweep.log
PREV=/tmp/ablsweep_seen.$$
: > "$PREV"
while true; do
  cur=$( {
    grep -hE "SWEEP_|REEVAL_DONE|exit=" "$SWEEP" 2>/dev/null
    for f in outputs/generated/ablation_gsm2b_att7_heldout_maxsteps*/run.log; do
      [ -f "$f" ] || continue
      lbl=$(basename "$(dirname "$f")")
      grep -hE "Accuracy:|Syntax:|out of memory|OutOfMemoryError|CUDA error|CUDA out|Traceback" "$f" 2>/dev/null \
        | sed "s#^#[$lbl] #"
    done
  } )
  comm -13 <(sort "$PREV") <(printf '%s\n' "$cur" | sort)
  printf '%s\n' "$cur" > "$PREV"
  if grep -q "SWEEP_ALL_DONE" "$SWEEP" 2>/dev/null; then echo "MONITOR: SWEEP_ALL_DONE"; break; fi
  if grep -qiE "out of memory|OutOfMemoryError|CUDA out" outputs/generated/ablation_gsm2b_att7_heldout_maxsteps*/run.log 2>/dev/null; then echo "MONITOR: OOM DETECTED"; break; fi
  sleep 30
done
rm -f "$PREV"
