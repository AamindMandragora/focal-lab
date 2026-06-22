#!/bin/bash
# Auto-chained synthesis launcher for 2026-06-04:
#   1. Waits for GSM-7B a21 reeval (PID_A21) to finish, then launches Spider-7B synthesis.
#   2. Waits for Spider-1.5B reeval (PID_SPIDER) to finish, then launches GSM-1.5B synthesis.
# Both waits run in parallel so the longer one doesn't block the shorter.
set -u

PID_A21="${1:-3763451}"       # GSM-7B a21 heldout reeval
PID_SPIDER="${2:-3850443}"    # Spider-1.5B a14 heldout reeval
cd /home/aadivyar/csd-generation

wait_and_launch_spider7b() {
    echo "[chain] Waiting for GSM-7B a21 reeval (PID $PID_A21) to finish..."
    while kill -0 "$PID_A21" 2>/dev/null; do sleep 10; done
    echo "[chain] GSM-7B a21 reeval done. Launching Spider-7B synthesis on GPU 1..."
    chmod +x launch_spider7b_disjoint_iter30_20260604.sh
    bash launch_spider7b_disjoint_iter30_20260604.sh
}

wait_and_launch_gsm1p5b() {
    echo "[chain] Waiting for Spider-1.5B reeval (PID $PID_SPIDER) to finish..."
    while kill -0 "$PID_SPIDER" 2>/dev/null; do sleep 10; done
    echo "[chain] Spider-1.5B reeval done. Launching GSM-1.5B synthesis on GPU 3..."
    chmod +x launch_gsm1p5b_disjoint_iter30_20260604.sh
    bash launch_gsm1p5b_disjoint_iter30_20260604.sh
}

wait_and_launch_spider7b >> /tmp/chain_spider7b_20260604.log 2>&1 &
echo "Spider-7B chain watcher PID=$! (log=/tmp/chain_spider7b_20260604.log)"

wait_and_launch_gsm1p5b >> /tmp/chain_gsm1p5b_20260604.log 2>&1 &
echo "GSM-1.5B chain watcher PID=$! (log=/tmp/chain_gsm1p5b_20260604.log)"
