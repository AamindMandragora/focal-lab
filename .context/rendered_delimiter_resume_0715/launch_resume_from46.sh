#!/usr/bin/env bash
set -euo pipefail

export REPO=/home/aadivyar/csd-generation
export RESUME_LAST_ATTEMPT=46
export RESUME_TOTAL_CAP=80
export RESUME_GPU=2
export RESUME_OUTPUT_NAME=warmfix_gsm14b_0714_r2
export RESUME_SEED_FILE="$REPO/.context/rendered_delimiter_resume_0715/gsm14b_attempt46.dfy"
export RESUME_HISTORY_FILE="$REPO/.context/rendered_delimiter_resume_0715/gsm14b_before46.json"
export RESUME_LOG_FILE="$REPO/logs/paid_synth_warmfix_gsm14b_0714_r2.log"

exec bash "$REPO/.context/resume_http429_cells.sh" worker gsm14b
