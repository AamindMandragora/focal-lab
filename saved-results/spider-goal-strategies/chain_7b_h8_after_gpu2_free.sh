#!/usr/bin/env bash
# Waits until GPU 2 has enough free memory for the 7B (the 1.5B 300-train re-eval finishing frees it),
# then launches the 7B H8 (iter40) synthesis run on GPU 2. Polls every 20s, gives up after ~40 min.
# 7B at util 0.45 needs ~18.4 GB free; require >=19000 MiB to be safe.
set -u
cd ~/csd-generation
CHAINLOG=logs/chain_7b_h8_$(date +%Y%m%d_%H%M%S).log
NEED_MIB=19000
GPU=2
MAX_TRIES=120   # 120 * 20s = 40 min cap
echo "[chain] waiting for GPU $GPU free >= ${NEED_MIB} MiB to launch 7B H8 (cap 40min)" | tee -a "$CHAINLOG"
i=0
while [ "$i" -lt "$MAX_TRIES" ]; do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')
  echo "[chain] try $i: GPU $GPU free=${FREE} MiB (need ${NEED_MIB})" | tee -a "$CHAINLOG"
  if [ -n "$FREE" ] && [ "$FREE" -ge "$NEED_MIB" ]; then
    echo "[chain] GPU $GPU has ${FREE} MiB free -> launching 7B H8 now" | tee -a "$CHAINLOG"
    bash saved-results/spider-goal-strategies/run_iter50_tok0_h8_iter40_7b.sh
    echo "[chain] 7B H8 launcher returned exit=$? at $(date -u)" | tee -a "$CHAINLOG"
    exit 0
  fi
  i=$((i+1))
  sleep 20
done
echo "[chain] GAVE UP after 40 min — GPU $GPU never freed ${NEED_MIB} MiB. 7B H8 NOT launched." | tee -a "$CHAINLOG"
exit 1
