#!/usr/bin/env bash
# Waits until GPU 1 or 2 has >=19000 MiB free, then launches the 7B 300-train iter40 run on it.
# Polls every 30s, cap ~8h (covers the 1.5B 300-train run finishing and freeing GPU 2, or GPU 1's
# other job ending). Only considers GPUs 1 and 2 — project rule: GPUs 0 and 3 are others' territory.
# The >=19000 free check prevents collision with the 1.5B run still on GPU 2.
set -u
cd ~/csd-generation
CHAINLOG=logs/chain_7b_300train_$(date +%Y%m%d_%H%M%S).log
NEED_MIB=19000
MAX_TRIES=960   # 960 * 30s = 8h cap
echo "[chain] waiting for GPU 1 or 2 free >= ${NEED_MIB} MiB to launch 7B 300-train (cap 8h)" | tee -a "$CHAINLOG"
i=0
while [ "$i" -lt "$MAX_TRIES" ]; do
  for G in 1 2; do
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$G" 2>/dev/null | tr -d ' ')
    echo "[chain] try $i: GPU $G free=${FREE} MiB (need ${NEED_MIB})" | tee -a "$CHAINLOG"
    if [ -n "$FREE" ] && [ "$FREE" -ge "$NEED_MIB" ]; then
      echo "[chain] GPU $G has ${FREE} MiB free -> launching 7B 300-train now on GPU $G" | tee -a "$CHAINLOG"
      GPU=$G bash saved-results/spider-goal-strategies/run_300train_cold_iter40_7b.sh
      echo "[chain] 7B 300-train launcher returned exit=$? at $(date -u)" | tee -a "$CHAINLOG"
      exit 0
    fi
  done
  i=$((i+1))
  sleep 30
done
echo "[chain] GAVE UP after 8h — no GPU freed ${NEED_MIB} MiB. 7B 300-train NOT launched." | tee -a "$CHAINLOG"
exit 1
