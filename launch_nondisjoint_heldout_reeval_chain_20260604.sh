#!/bin/bash
# Sequential chain: run all 4 non-disjoint held-out reevals on GPU 2.
# Order: GSM-1.5B → GSM-7B → Spider-1.5B → Spider-7B
# Each waits for the previous to finish before starting.
set -u
cd /home/aadivyar/csd-generation

run_step() {
    local label="$1"
    local script="$2"
    echo "[chain] Starting $label..."
    chmod +x "$script"
    bash "$script"
    echo "[chain] $label done."
}

run_step "GSM-1.5B nondisjoint heldout reeval" launch_gsm1p5b_nondisjoint_heldout_reeval_20260604.sh
run_step "GSM-7B nondisjoint heldout reeval"   launch_gsm7b_nondisjoint_heldout_reeval_20260604.sh
run_step "Spider-1.5B nondisjoint heldout reeval" launch_spider1p5b_nondisjoint_heldout_reeval_20260604.sh
run_step "Spider-7B nondisjoint heldout reeval"   launch_spider7b_nondisjoint_heldout_reeval_20260604.sh

echo "[chain] All 4 nondisjoint heldout reevals complete."
