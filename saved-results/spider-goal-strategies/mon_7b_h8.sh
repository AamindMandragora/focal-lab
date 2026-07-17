#!/usr/bin/env bash
# Poll-loop monitor (runs ON focal, streamed over ssh). Emits a combined status line ONLY when the
# state changes; exits when BOTH the 1.5B 300-train re-eval AND the 7B H8 run reach a terminal state.
# Covers happy path + failure: re-eval done, chain waiting/launched/gave-up, 7B H8 win/error/finished-no-win.
cd ~/csd-generation
prev=""
for i in $(seq 1 960); do   # 8h cap at 30s/tick
  reev_done=0; h7_done=0
  # --- 1.5B 300-train re-eval ---
  RDIR=$(ls -dt outputs/generated/*spider1p5b_h8win_reeval_300train*/results 2>/dev/null | head -1)
  if [ -n "$RDIR" ] && [ -f "$RDIR/success_report.json" ]; then
    RA=$(python3 - "$RDIR/success_report.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); e=d.get('evaluation_result',{})
print("%s/%s=%.4f syn=%s es=%s"%(e.get('num_correct'),e.get('num_examples'),(e.get('accuracy') or 0),e.get('syntax_rate'),e.get('early_stopped')))
PY
)
    reev="REEVAL_DONE[$RA]"; reev_done=1
  else
    RL=$(ls -t logs/spider1p5b_h8win_reeval_300train*.log 2>/dev/null | head -1)
    P=$(grep -oE "Processing example [0-9]+/[0-9]+" "$RL" 2>/dev/null | tail -1)
    reev="reeval[$P]"
  fi
  # --- chain + 7B H8 ---
  CL=$(ls -t logs/chain_7b_h8_*.log 2>/dev/null | head -1)
  if grep -q "launching 7B H8 now" "$CL" 2>/dev/null; then
    HL=$(ls -t logs/spider7b_iter50_tok0_h8_iter40_cold_*.log 2>/dev/null | head -1)
    HDIR=$(ls -dt outputs/generated/*spider7b_iter50_tok0_h8_iter40*/results 2>/dev/null | head -1)
    HA=$(grep -E "Accuracy:|accuracy=" "$HL" 2>/dev/null | tail -1 | tr -s ' ' | cut -c1-70)
    if [ -n "$HDIR" ] && [ -f "$HDIR/success_report.json" ]; then
      h7="7B_H8_WIN(success_report present)"; h7_done=1
    elif grep -qE "Traceback|CUDA out of memory|Killed|RuntimeError" "$HL" 2>/dev/null; then
      h7="7B_H8_ERROR(check $HL)"; h7_done=1
    elif grep -qhE "\[done\].*exit=" logs/iter50_tok0_h8_7b_driver_*.log 2>/dev/null; then
      h7="7B_H8_FINISHED_NO_WIN(driver exited, no 0.76 accept)"; h7_done=1
    else
      h7="7b_h8_running[$HA]"
    fi
  elif grep -q "GAVE UP" "$CL" 2>/dev/null; then
    h7="7B_H8_NOT_LAUNCHED(gpu never freed in 40min)"; h7_done=1
  else
    CF=$(grep -oE "free=[0-9]+ MiB" "$CL" 2>/dev/null | tail -1)
    h7="chain_waiting[$CF]"
  fi
  cur="$reev || $h7"
  if [ "$cur" != "$prev" ]; then echo "$cur"; prev="$cur"; fi
  if [ "$reev_done" -eq 1 ] && [ "$h7_done" -eq 1 ]; then echo "BOTH_TERMINAL — monitor exiting"; exit 0; fi
  sleep 30
done
echo "monitor cap reached (8h) — exiting"
