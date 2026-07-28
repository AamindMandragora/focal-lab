#!/usr/bin/env bash
set -euo pipefail

repo=/home/aadivyar/csd-generation
python=/apps/conda/aadivyar/envs/csd/bin/python
state_dir="$repo/.context/parallel_crane_gpu0"
pid_file="$state_dir/worker.pid"
exit_file="$state_dir/worker.exit"
meta_file="$state_dir/worker.meta"
log_file="$repo/logs/focal_collection_spider_9b_crane.log"
output_json="$repo/outputs/baselines/crane/Qwen_Qwen3-5-9B/spider_seed334_test300__tb1__ms600.json"
claim_dir="${output_json}.running"
worker_pid=""
worker_pgid=""

mkdir -p "$state_dir" "$(dirname "$output_json")" "$(dirname "$log_file")"
if [ -s "$output_json" ]; then
  echo "CRANE output already exists: $output_json" >&2
  exit 0
fi
if ! mkdir "$claim_dir"; then
  echo "CRANE output is already claimed: $claim_dir" >&2
  exit 1
fi
cleanup_claim() {
  if process_group_alive; then
    printf '%s [parallel-crane] retaining_claim=%s live_pgid=%s\n' \
      "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$claim_dir" "$worker_pgid" >> "$log_file"
    return
  fi
  rmdir "$claim_dir" 2>/dev/null || true
}
process_group_alive() {
  if [ -z "$worker_pgid" ]; then
    candidate_pid=${worker_pid:-$!}
    if [ -n "$candidate_pid" ] && [ -r "/proc/$candidate_pid/stat" ]; then
      worker_pgid=$(ps -o pgid= -p "$candidate_pid" | tr -d '[:space:]')
    fi
  fi
  [ -n "$worker_pgid" ] && kill -0 -- "-$worker_pgid" 2>/dev/null
}
handle_signal() {
  cleanup_claim
  exit 130
}
trap cleanup_claim EXIT
trap handle_signal INT TERM
rm -f "$pid_file" "$exit_file" "$meta_file"

cd "$repo"
export CUDA_VISIBLE_DEVICES=0
export CSD_HF_KV_CACHE=0
export PYTHONUNBUFFERED=1
export PYTHONPATH=synthesis/evaluate:.

printf '%s [parallel-crane] launch gpu=0 output=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$output_json" >> "$log_file"
setsid "$python" -m synthesis.evaluate.run_legacy_fixed_strategy \
  --strategy crane \
  --dataset spider \
  --eval-model Qwen/Qwen3.5-9B \
  --eval-backend vllm \
  --device cuda \
  --eval-sample-size 300 \
  --eval-max-steps 600 \
  --eval-step-token-budget 1 \
  --spider-split-file environment/benchmark_splits/spider_dev_proportional_300x300_seed334.json \
  --spider-split-name test \
  --vllm-gpu-memory-utilization 0.90 \
  --vllm-tensor-parallel-size 1 \
  --output-json "$output_json" >> "$log_file" 2>&1 &
worker_pid=$!
printf '%s\n' "$worker_pid" > "$pid_file"
worker_start=$(awk '{print $22}' "/proc/$worker_pid/stat")
worker_pgid=$(ps -o pgid= -p "$worker_pid" | tr -d '[:space:]')
printf '%s %s %s\n' "$worker_pid" "$worker_start" "$worker_pgid" > "$meta_file"

set +e
wait "$worker_pid"
exit_code=$?
set -e
printf '%s\n' "$exit_code" > "$exit_file"
printf '%s [parallel-crane] exit=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$exit_code" >> "$log_file"
exit "$exit_code"
