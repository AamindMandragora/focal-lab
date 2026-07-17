#!/usr/bin/env bash
set -euo pipefail

GPU_INDEX=2
MAX_USED_MIB=3500
POLL_SECONDS=15
LAUNCHER=/home/aadivyar/csd-generation/.context/rendered_delimiter_resume_0715/launch_resume_from46.sh

while true; do
  used_mib=$(
    nvidia-smi --id="$GPU_INDEX" --query-gpu=memory.used --format=csv,noheader,nounits \
      2>/dev/null | tr -d ' '
  ) || used_mib=""
  if [[ "$used_mib" =~ ^[0-9]+$ ]] && (( used_mib <= MAX_USED_MIB )); then
    printf '%s [gsm14b-rendered-resume] gpu=%d used_mib=%d launching\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$GPU_INDEX" "$used_mib"
    exec "$LAUNCHER"
  fi
  printf '%s [gsm14b-rendered-resume] gpu=%d used_mib=%s waiting\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$GPU_INDEX" "${used_mib:-unknown}"
  sleep "$POLL_SECONDS"
done
