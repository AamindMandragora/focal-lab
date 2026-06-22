#!/bin/bash
set -u

cd /home/aadivyar/csd-generation

LOG=/tmp/gsm1p5b_fresh_watch_20260608.log

while true; do
  free_gpu=""
  while IFS=, read -r gpu_idx mem_used; do
    gpu_idx="${gpu_idx//[[:space:]]/}"
    mem_used="${mem_used//[[:space:]]/}"
    if [ -n "$gpu_idx" ] && [ "${mem_used:-0}" -lt 1000 ]; then
      free_gpu="$gpu_idx"
      break
    fi
  done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)

  if [ -n "$free_gpu" ]; then
    echo "$(date -Iseconds) launching fresh GSM-1.5B run on GPU $free_gpu" >> "$LOG"
    CUDA_VISIBLE_DEVICES="$free_gpu" ./launch_gsm1p5b_fresh_20260608.sh >> "$LOG" 2>&1
    exit $?
  fi

  echo "$(date -Iseconds) no free GPU; sleeping" >> "$LOG"
  sleep 120
done
