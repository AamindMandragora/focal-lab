#!/usr/bin/env bash
set -euo pipefail

scheduler_pid=${1:?scheduler PID is required}
repo=/home/aadivyar/csd-generation
state_dir="$repo/.context/parallel_crane_gpu0"
pid_file="$state_dir/worker.pid"
exit_file="$state_dir/worker.exit"
meta_file="$state_dir/worker.meta"
guard_log="$repo/logs/focal_parallel_crane_gpu0_guard.log"
output_json="$repo/outputs/baselines/crane/Qwen_Qwen3-5-9B/spider_seed334_test300__tb1__ms600.json"
claim_dir="${output_json}.running"
memory_limit_mib=38500
pid_wait_limit_seconds=60
shutdown_wait_limit_seconds=30
resumed=0
worker_pid=""
worker_start=""
worker_pgid=""

resume_scheduler() {
  if [ "$resumed" -eq 0 ] && kill -0 "$scheduler_pid" 2>/dev/null; then
    kill -CONT "$scheduler_pid"
    printf '%s [parallel-crane-guard] scheduler_resumed pid=%s\n' \
      "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$scheduler_pid" >> "$guard_log"
  fi
  resumed=1
}
trap resume_scheduler EXIT INT TERM

worker_identity_matches() {
  [ -n "$worker_pid" ] && [ -n "$worker_start" ] && [ -n "$worker_pgid" ] || return 1
  [ -r "/proc/$worker_pid/stat" ] || return 1
  current_start=$(awk '{print $22}' "/proc/$worker_pid/stat")
  current_pgid=$(ps -o pgid= -p "$worker_pid" | tr -d '[:space:]')
  current_command=$(tr '\0' ' ' < "/proc/$worker_pid/cmdline")
  [ "$current_start" = "$worker_start" ] || return 1
  [ "$current_pgid" = "$worker_pgid" ] || return 1
  case "$current_command" in
    *"run_legacy_fixed_strategy --strategy crane --dataset spider"*) return 0 ;;
    *) return 1 ;;
  esac
}

worker_group_alive() {
  [ -n "$worker_pgid" ] && kill -0 -- "-$worker_pgid" 2>/dev/null
}

verified_identity=0
for ((waited=0; waited<pid_wait_limit_seconds; waited++)); do
  if [ -s "$meta_file" ]; then
    read -r worker_pid worker_start worker_pgid < "$meta_file"
    if worker_identity_matches; then
      verified_identity=1
      break
    fi
  fi
  sleep 1
done
if [ "$verified_identity" -ne 1 ]; then
  printf '%s [parallel-crane-guard] worker_identity_timeout seconds=%s\n' \
    "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$pid_wait_limit_seconds" >> "$guard_log"
  exit 1
fi

stop_crane_group() {
  if worker_identity_matches; then
    kill -TERM -- "-$worker_pgid"
  else
    printf '%s [parallel-crane-guard] process_identity_changed; refusing_signal pid=%s\n' \
      "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$worker_pid" >> "$guard_log"
  fi
}

printf '%s [parallel-crane-guard] start worker_pid=%s scheduler_pid=%s limit_mib=%s\n' \
  "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$worker_pid" "$scheduler_pid" "$memory_limit_mib" >> "$guard_log"

while worker_identity_matches; do
  if ! used_mib=$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]'); then
    printf '%s [parallel-crane-guard] memory_query_failed; stopping_crane_pgid=%s\n' \
      "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$worker_pgid" >> "$guard_log"
    stop_crane_group
    break
  fi
  if [ "$used_mib" -gt "$memory_limit_mib" ]; then
    printf '%s [parallel-crane-guard] memory_limit_exceeded used_mib=%s; stopping_crane_pgid=%s\n' \
      "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$used_mib" "$worker_pgid" >> "$guard_log"
    stop_crane_group
    break
  fi
  sleep 2
done

shutdown_complete=0
for ((waited=0; waited<shutdown_wait_limit_seconds; waited++)); do
  if ! worker_group_alive; then
    shutdown_complete=1
    break
  fi
  sleep 1
done
if [ "$shutdown_complete" -ne 1 ]; then
  printf '%s [parallel-crane-guard] shutdown_wait_timeout seconds=%s; retaining_claim=%s\n' \
    "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$shutdown_wait_limit_seconds" "$claim_dir" >> "$guard_log"
fi
if ! worker_group_alive; then
  rmdir "$claim_dir" 2>/dev/null || true
fi
resume_scheduler
trap - EXIT INT TERM
