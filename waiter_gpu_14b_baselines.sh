#!/bin/bash
# GPU-memory waiter for the 14B baseline sweep.
# Polls nvidia-smi every 5 minutes; when any single GPU has >= 34000 MiB free,
# logs the event, sets GPU=<idx>, and exec's the master script in place of this
# process (so the master script inherits the same nohup/log context).
#
# Inputs:  none (reads nvidia-smi at runtime)
# Outputs: logs to /home/aadivyar/csd-generation/waiter_gpu_14b_baselines.log,
#          then exec's run_14b_baselines_20260612.sh with GPU exported.
#
# Algorithm:
#   1. Every 5 minutes call nvidia-smi to list free MiB per GPU.
#   2. If nvidia-smi fails (transient error), log a warning and loop — never crash.
#   3. For each GPU (0..N), if free >= FREE_THRESHOLD, record it and break.
#   4. On first qualifying GPU: log GPU_FREE_DETECTED, export GPU=<idx>, exec master script.
#
# 34000 MiB threshold: 14B weights are ~28 GB; vllm at 0.85 util needs ~34 GB.
# The three stale processes (774909/1262510/3042774) each hold 10-36 GB; once
# the user kills them any of GPU 0/2/3 will cross this threshold.

MASTERSCRIPT=/home/aadivyar/csd-generation/run_14b_baselines_20260612.sh
LOGFILE=/home/aadivyar/csd-generation/waiter_gpu_14b_baselines.log
FREE_THRESHOLD=34000   # MiB

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGFILE"; }

log "WAITER_START pid=$$ threshold=${FREE_THRESHOLD}MiB master=$MASTERSCRIPT"

while true; do
    # --- query nvidia-smi (failure-tolerant) ---
    SMIOUT=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>&1)
    SMIEXIT=$?
    if [ "$SMIEXIT" -ne 0 ]; then
        log "WARNING nvidia-smi returned $SMIEXIT — output: $SMIOUT — retrying in 60s"
        sleep 60
        continue
    fi

    # --- scan GPUs ---
    CHOSEN_GPU=""
    CHOSEN_FREE=""
    while IFS=', ' read -r idx free; do
        # strip whitespace
        idx="${idx// /}"
        free="${free// /}"
        # skip non-numeric lines
        case "$free" in
            ''|*[!0-9]*) continue ;;
        esac
        log "POLL gpu=$idx free=${free}MiB threshold=${FREE_THRESHOLD}MiB"
        if [ "$free" -ge "$FREE_THRESHOLD" ]; then
            CHOSEN_GPU="$idx"
            CHOSEN_FREE="$free"
            break
        fi
    done <<< "$SMIOUT"

    if [ -n "$CHOSEN_GPU" ]; then
        log "GPU_FREE_DETECTED gpu=$CHOSEN_GPU free=${CHOSEN_FREE}MiB $(date)"
        log "LAUNCHING master script: $MASTERSCRIPT"
        export GPU="$CHOSEN_GPU"
        # exec replaces this process — master script output continues into the
        # same nohup log (MASTERLOG inside the script is separate, per-event).
        exec bash "$MASTERSCRIPT"
        # if exec somehow fails (e.g. script not found), fall through and log
        log "ERROR exec failed (exit $?) — master script not launched"
        exit 1
    fi

    sleep 300   # 5 minutes
done
