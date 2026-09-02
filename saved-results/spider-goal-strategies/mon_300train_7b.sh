#!/usr/bin/env bash
# Poll-loop monitor for the 7B 300-train iter40 run (launched by chain_7b_300train.sh once a GPU frees).
# Emits a status line ONLY on state change; exits on terminal. Covers: chain waiting / launched /
# gave-up, and 7B win / finished-no-accept / error. Per-attempt granularity (attempt count), NOT
# per-example, so it does not flood.
cd ~/csd-generation
prev=""
for i in $(seq 1 1200); do   # 10h cap at 30s/tick
  CL=$(ls -t logs/chain_7b_300train_*.log 2>/dev/null | head -1)
  HL=$(ls -t logs/spider7b_300train_cold_iter40_*.log 2>/dev/null | head -1)
  HDIR=$(ls -dt outputs/generated/*spider7b_300train_cold_iter40*/results 2>/dev/null | head -1)
  ATT=$(grep -E "Accuracy:" "$HL" 2>/dev/null | tail -1 | tr -s ' ' | cut -c1-70)
  ATTN=$(grep -cE "Accuracy:" "$HL" 2>/dev/null)
  done=0
  if [ -n "$HDIR" ] && [ -f "$HDIR/success_report.json" ]; then
    st="7B_300TRAIN_ACCEPTED(success_report present) [$ATT]"; done=1
  elif [ -n "$HL" ] && grep -qE "Traceback|CUDA out of memory|Killed|RuntimeError" "$HL" 2>/dev/null; then
    st="7B_300TRAIN_ERROR(check $HL)"; done=1
  elif grep -qhE "\[done\].*exit=" logs/300train_7b_driver_*.log 2>/dev/null; then
    st="7B_300TRAIN_FINISHED_NO_ACCEPT(ran out of 40 iters; take best from failure_report) [$ATT]"; done=1
  elif grep -q "GAVE UP" "$CL" 2>/dev/null; then
    st="7B_300TRAIN_NOT_LAUNCHED(no GPU freed in 8h)"; done=1
  elif grep -q "launching 7B 300-train now" "$CL" 2>/dev/null; then
    st="7b_300train_running[attempt#$ATTN done | latest:$ATT]"
  else
    st="7b_chain_waiting(polling GPU 1/2 for >=19GB free)"
  fi
  if [ "$st" != "$prev" ]; then echo "$st"; prev="$st"; fi
  if [ "$done" -eq 1 ]; then echo "TERMINAL — monitor exiting"; exit 0; fi
  sleep 30
done
echo "monitor cap reached (10h) — exiting"
