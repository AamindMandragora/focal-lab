#!/usr/bin/env bash
# Explicit user-approved warm continuation for two Bedrock-429-interrupted GSM
# cells. Each last strategy is replayed so evaluation feedback is reconstructed
# before a new author call.
set -uo pipefail

REPO="${REPO:-/home/aadivyar/csd-generation}"
RUNNER="$REPO/.context/resume_http429_cells.sh"
LOG="$REPO/logs/deferred_gsm_resume_controller.log"
STATE="$REPO/.context/deferred_gsm_resume_state.tsv"
POLL_SECONDS="${POLL_SECONDS:-30}"

mkdir -p "$REPO/logs" "$REPO/.context"

log() {
  printf '%s [deferred-gsm] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG"
}

gpu_used_mib() {
  nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' '
}

compiled_csd_for_run() {
  "$REPO/.context/resolve_success_csd.py" "$1"
}

run_heldout() {
  local cell="$1" gpu="$2" csd model util output_json
  case "$cell" in
    gsm-qwen35-9b)
      model=Qwen/Qwen3.5-9B; util=0.60
      output_json="$REPO/outputs/controlled_comparison/post14b_rebar/gsm-qwen35-9b.json"
      ;;
    gsm14b)
      model=Qwen/Qwen2.5-14B-Instruct; util=0.81
      output_json="$REPO/outputs/controlled_comparison/gsm_14B/metadecode_paid0708_infraretry_kvfix_0711.json"
      ;;
    *) log "HELDOUT_CONFIG_MISSING cell=$cell"; return 2 ;;
  esac
  if [[ -s "$output_json" ]]; then
    log "HELDOUT_SKIP_EXISTS cell=$cell output=$output_json"
    return 0
  fi
  csd=$(compiled_csd_for_run "$cell") || {
    log "HELDOUT_NO_SUCCESS_CSD cell=$cell"
    return 1
  }
  mkdir -p "$(dirname "$output_json")"
  log "HELDOUT_START cell=$cell gpu=$gpu csd=$csd output=$output_json"
  CUDA_VISIBLE_DEVICES="$gpu" /apps/conda/aadivyar/envs/csd/bin/python \
    -m synthesis.scripts.reevaluate_compiled_csd "$csd" \
    --dataset gsm_symbolic --eval-model "$model" --eval-backend vllm --device auto \
    --sample-size 49 --max-steps 900 --step-token-budget 1 \
    --max-seconds-per-example 600 --vllm-gpu-memory-utilization "$util" \
    --vllm-tensor-parallel-size 1 \
    --gsm-split-file environment/benchmark_splits/gsm_symbolic_crane_proportional_49x49_seed123.json \
    --gsm-split-name eval --output-json "$output_json" >>"$LOG" 2>&1
  local status=$?
  log "HELDOUT_FINISH cell=$cell exit=$status output=$output_json"
  return "$status"
}

wait_then_launch() {
  local cell="$1" watched_pid="$2" gpu="$3" max_used="$4" attempt="$5" source_log="$6" history_file="$7"
  local used wrapper status

  log "WAIT cell=$cell after_pid=$watched_pid gpu=$gpu max_used_mib=$max_used"
  while kill -0 "$watched_pid" 2>/dev/null; do sleep "$POLL_SECONDS"; done
  while true; do
    used=$(gpu_used_mib "$gpu")
    if [[ "$used" =~ ^[0-9]+$ ]] && (( used <= max_used )); then break; fi
    log "GPU_WAIT cell=$cell gpu=$gpu used_mib=${used:-unknown} max_used_mib=$max_used"
    sleep "$POLL_SECONDS"
  done

  if [[ "${DRY_RUN:-0}" == 1 ]]; then
    log "DRY_RUN_READY cell=$cell gpu=$gpu attempt=$attempt source=$source_log history=$history_file"
    return 0
  fi

  printf '%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$cell" launching "gpu=$gpu replay_attempt=$attempt" >>"$STATE"
  RESUME_GPU="$gpu" RESUME_LAST_ATTEMPT="$attempt" RESUME_SOURCE_LOG="$source_log" RESUME_HISTORY_FILE="$history_file" \
    "$RUNNER" worker "$cell" >>"$LOG" 2>&1 &
  wrapper=$!
  log "LAUNCHED cell=$cell wrapper_pid=$wrapper gpu=$gpu replay_attempt=$attempt"
  wait "$wrapper"; status=$?
  printf '%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$cell" finished "exit=$status" >>"$STATE"
  log "FINISHED cell=$cell wrapper_pid=$wrapper exit=$status"
  if (( status == 0 )); then
    run_heldout "$cell" "$gpu"
    status=$?
    printf '%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$cell" heldout_finished "exit=$status" >>"$STATE"
  else
    log "HELDOUT_SKIP_NO_SYNTHESIS_WIN cell=$cell synthesis_exit=$status"
  fi
  return "$status"
}

if [[ ! -f "$STATE" ]]; then
  printf 'timestamp\tcell\tstatus\tdetail\n' >"$STATE"
fi

log "CONTROLLER_START billing_account=887730490125 region=us-east-1 warm_override=gsm9-attempt5,gsm14b-attempt33"
wait_then_launch gsm-qwen35-9b 4117130 1 14000 5 \
  "$REPO/logs/paid_synth_http429_resume_gsm-qwen35-9b.log" \
  "$REPO/.context/http429_resume_seeds/gsm-qwen35-9b_history_before5.json" &
gsm9_waiter=$!
wait_then_launch gsm14b 2713764 2 6000 33 \
  "$REPO/logs/paid_synth_infra_retry_gsm14b.log" \
  "$REPO/.context/http429_resume_seeds/gsm14b_history_before33.json" &
gsm14_waiter=$!
log "WAITERS gsm9=$gsm9_waiter gsm14b=$gsm14_waiter"
wait "$gsm9_waiter"; gsm9_status=$?
wait "$gsm14_waiter"; gsm14_status=$?
log "CONTROLLER_DONE gsm9_exit=$gsm9_status gsm14b_exit=$gsm14_status"
