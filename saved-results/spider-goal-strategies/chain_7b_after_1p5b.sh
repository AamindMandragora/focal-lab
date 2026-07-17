#!/usr/bin/env bash
# Durable chainer: wait until the live 1.5B cycle-1 cold run finishes (frees GPU 2),
# then launch the 7B cycle-1 cold run. Run with nohup ON focal so it survives SSH
# disconnects across the multi-hour 1.5B run. Both cells were user-approved
# ("both cells" 2026-06-22); same work Bedrock account (AWS_BEARER_TOKEN_BEDROCK,
# us-east-1) the 1.5B run already spends on.
set -u
cd ~/csd-generation
MARK="run_synthesis.*spider1p5b_cycle1_token0_cold_20260623_072721"
SEVENB="saved-results/spider-goal-strategies/run_cycle1_cold_7b.sh"
CLOG="logs/chain_7b_after_1p5b_$(date +%Y%m%d_%H%M%S).log"

echo "[chainer] start $(date -u): waiting for 1.5B ($MARK) to finish" | tee -a "$CLOG"
while pgrep -f "$MARK" >/dev/null 2>&1; do
  sleep 120
done
echo "[chainer] 1.5B gone at $(date -u); launching 7B" | tee -a "$CLOG"

nohup bash -l "$SEVENB" > "logs/cycle1_7b_launcher_$(date +%Y%m%d_%H%M%S).out" 2>&1 &
sleep 10
if pgrep -af "run_synthesis.*spider7b" >/dev/null 2>&1; then
  echo "[chainer] 7B launched OK at $(date -u)" | tee -a "$CLOG"
  pgrep -af "run_synthesis.*spider7b" | tee -a "$CLOG"
else
  echo "[chainer] WARN 7B not visible after launch at $(date -u)" | tee -a "$CLOG"
fi
echo "[chainer] done $(date -u)" | tee -a "$CLOG"
