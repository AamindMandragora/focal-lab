#!/usr/bin/env bash
# Poll-loop monitor (runs ON focal, streamed over ssh) for the 1.5B 300-TRAIN cold iter40 run.
# Emits a status line ONLY when state changes; exits on terminal. Covers win + finished-no-accept + error.
cd ~/csd-generation
prev=""
for i in $(seq 1 1200); do   # 10h cap at 30s/tick (300-eval/attempt is slow)
  HL=$(ls -t logs/spider1p5b_300train_cold_iter40_*.log 2>/dev/null | head -1)
  HDIR=$(ls -dt outputs/generated/*spider1p5b_300train_cold_iter40*/results 2>/dev/null | head -1)
  ATT=$(grep -E "Accuracy:" "$HL" 2>/dev/null | tail -1 | tr -s ' ' | cut -c1-70)
  ATTN=$(grep -cE "Accuracy:" "$HL" 2>/dev/null)   # attempt count = how many evals finished
  done=0
  if [ -n "$HDIR" ] && [ -f "$HDIR/success_report.json" ]; then
    st="300TRAIN_ACCEPTED(success_report present) [$ATT]"; done=1
  elif grep -qE "Traceback|CUDA out of memory|Killed|RuntimeError" "$HL" 2>/dev/null; then
    st="300TRAIN_ERROR(check $HL)"; done=1
  elif grep -qhE "\[done\].*exit=" logs/300train_1p5b_driver_*.log 2>/dev/null; then
    st="300TRAIN_FINISHED_NO_ACCEPT(ran out of 40 iters; take best from failure_report) [$ATT]"; done=1
  else
    st="300train_running[attempt#$ATTN done | latest:$ATT]"
  fi
  if [ "$st" != "$prev" ]; then echo "$st"; prev="$st"; fi
  if [ "$done" -eq 1 ]; then echo "TERMINAL — monitor exiting"; exit 0; fi
  sleep 30
done
echo "monitor cap reached (10h) — exiting"
