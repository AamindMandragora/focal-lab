#!/usr/bin/env bash
# Explicit user-approved exception (2026-07-11): resume four runs interrupted by
# Bedrock HTTP 429 by replaying their last completed attempt, then rebuilding the
# normal evaluation-failure refinement prompt. This does not change the default
# cold-only launcher or any other synthesis run.
set -uo pipefail

REPO="${REPO:-/home/aadivyar/csd-generation}"
PY="${PY:-/apps/conda/aadivyar/envs/csd/bin/python}"
MODE="${1:-controller}"
CELL="${2:-}"
LOG_DIR="$REPO/logs"
RECOVERY_DIR="$REPO/.context/http429_resume_seeds"
QUEUE_LOG="$LOG_DIR/paid_synth_http429_resume_queue_driver.log"
LOCK_DIR="${HTTP429_LOCK_DIR:-$REPO/.context/http429_resume_locks}"
FLOCK_BIN="${FLOCK_BIN:-flock}"
PROC_ROOT="${HTTP429_PROC_ROOT:-/proc}"
POLL_SECONDS="${POLL_SECONDS:-30}"

log() {
  printf '%s [http429-resume] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

output_name_for_cell() {
  if [[ -n "${RESUME_OUTPUT_NAME:-}" ]]; then
    printf '%s\n' "$RESUME_OUTPUT_NAME"
    return 0
  fi
  case "$1" in
    gsm-qwen35-2b) printf '%s\n' post14b_rebar_gsm-qwen35-2b_0711 ;;
    gsm-qwen35-4b) printf '%s\n' post14b_rebar_gsm-qwen35-4b_0711 ;;
    gsm-qwen35-9b) printf '%s\n' post14b_rebar_gsm-qwen35-9b_0711 ;;
    smiles-qwen35-9b-isocyanates) printf '%s\n' post14b_rebar_smiles-qwen35-9b-isocyanates_0711 ;;
    spider-qwen35-4b) printf '%s\n' post14b_rebar_spider-qwen35-4b_0711 ;;
    spider-qwen35-9b) printf '%s\n' post14b_rebar_spider-qwen35-9b_0711 ;;
    spider-qwen25-7b) printf '%s\n' post14b_rebar_spider-qwen25-7b_0711 ;;
    gsm14b) printf '%s\n' synth_gsm14b_z3bar_retry_0708_infraretry_kvfix_0711 ;;
    *) return 2 ;;
  esac
}

active_worker_records() {
  local cell="$1" output_name
  output_name=$(output_name_for_cell "$cell") || return 2
  "$PY" - "$PROC_ROOT" "$output_name" <<'PY'
from pathlib import Path
import sys

proc_root = Path(sys.argv[1])
target_output = sys.argv[2]

try:
    process_dirs = list(proc_root.iterdir())
except Exception as exc:
    print(f"cannot list {proc_root}: {exc}", file=sys.stderr)
    raise SystemExit(76)

matches = []
for process_dir in process_dirs:
    if not process_dir.name.isdigit():
        continue
    try:
        raw_args = process_dir.joinpath("cmdline").read_bytes()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        continue
    args = [part.decode("utf-8", "replace") for part in raw_args.split(b"\0") if part]
    is_synthesis = any(
        args[index:index + 2] == ["-m", "synthesis.run_synthesis"]
        for index in range(len(args) - 1)
    )
    output_names = [
        args[index + 1]
        for index, argument in enumerate(args[:-1])
        if argument == "--output-name"
    ]
    if not is_synthesis or target_output not in output_names:
        continue
    try:
        stat_text = process_dir.joinpath("stat").read_text()
        stat_after_name = stat_text.rsplit(") ", 1)[1].split()
        start_ticks = int(stat_after_name[19])
    except (FileNotFoundError, ProcessLookupError):
        continue
    except Exception as exc:
        print(f"cannot inspect matching pid {process_dir.name}: {exc}", file=sys.stderr)
        raise SystemExit(76)
    matches.append((start_ticks, int(process_dir.name)))

for start_ticks, pid in sorted(matches):
    print(f"{start_ticks}\t{pid}")
PY
}

lookup_active_worker() {
  local result_name="$1" cell="$2" records status count pids pid
  records=$(active_worker_records "$cell" 2>&1)
  status=$?
  if (( status != 0 )); then
    log "PROCESS_LOOKUP_FAILED cell=$cell detail=$records"
    return 76
  fi
  if [[ -z "$records" ]]; then
    printf -v "$result_name" '%s' ''
    return 1
  fi
  count=$(printf '%s\n' "$records" | wc -l | tr -d ' ')
  if (( count != 1 )); then
    pids=$(printf '%s\n' "$records" | cut -f2 | paste -sd, -)
    log "MULTIPLE_ACTIVE cell=$cell pids=$pids"
    return 76
  fi
  pid=${records#*$'\t'}
  printf -v "$result_name" '%s' "$pid"
  return 0
}

acquire_cell_lock() {
  local lock_file="$LOCK_DIR/$CELL.lock"
  mkdir -p "$LOCK_DIR"
  exec 9>"$lock_file"
  if ! "$FLOCK_BIN" -n 9; then
    log "SKIP_DUPLICATE_LOCK cell=$CELL lock=$lock_file"
    return 75
  fi
}

acquire_controller_lock() {
  local lock_file="$LOCK_DIR/controller.lock"
  mkdir -p "$LOCK_DIR"
  exec 7>"$lock_file"
  if ! "$FLOCK_BIN" -n 7; then
    log "SKIP_DUPLICATE_CONTROLLER lock=$lock_file"
    return 75
  fi
}

acquire_waiter_lock() {
  local lock_file="$LOCK_DIR/waiter-smiles-qwen35-9b-isocyanates.lock"
  mkdir -p "$LOCK_DIR"
  exec 8>"$lock_file"
  if ! "$FLOCK_BIN" -n 8; then
    log "SKIP_DUPLICATE_WAITER lock=$lock_file"
    return 75
  fi
}

plan_worker() {
  local result_name="$1" cell="$2" active_pid='' lookup_status
  lookup_active_worker active_pid "$cell"
  lookup_status=$?
  case "$lookup_status" in
    0) printf -v "$result_name" '%s' "$active_pid" ;;
    1) printf -v "$result_name" '%s' '' ;;
    *) return "$lookup_status" ;;
  esac
}

launch_planned_worker() {
  local result_name="$1" cell="$2" planned_pid="$3" launched_pid
  if [[ -n "$planned_pid" ]]; then
    log "REUSE cell=$cell pid=$planned_pid reason=already_running"
    printf -v "$result_name" '%s' "$planned_pid"
  elif [[ "${DRY_RUN:-0}" == 1 ]]; then
    log "DRY_RUN_WOULD_LAUNCH cell=$cell"
    printf -v "$result_name" '%s' none
  else
    nohup "$0" worker "$cell" 7>&- >>"$QUEUE_LOG" 2>&1 &
    launched_pid=$!
    printf -v "$result_name" '%s' "$launched_pid"
    log "LAUNCHED cell=$cell pid=$launched_pid"
  fi
}

extract_last_strategy() {
  local source_log="$1" attempt="$2" destination="$3"
  "$PY" - "$source_log" "$attempt" "$destination" <<'PY'
from pathlib import Path
import sys

source, attempt, destination = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
text = source.read_text(encoding="utf-8", errors="replace")
marker = f"Attempt {attempt}/40"
start = text.rfind(marker)
if start < 0:
    raise SystemExit(f"missing {marker} in {source}")
strategy_start = text.find("Strategy: ", start)
stage_start = text.find("\n\n[1/4] Verifying with Dafny...", strategy_start)
if strategy_start < 0 or stage_start < 0:
    raise SystemExit(f"could not isolate strategy for attempt {attempt} in {source}")
strategy = text[strategy_start + len("Strategy: "):stage_start].strip() + "\n"
if "method Main" in strategy or len(strategy) < 100:
    raise SystemExit(f"invalid recovered method body for attempt {attempt}")
destination.write_text(strategy, encoding="utf-8")
print(f"recovered attempt={attempt} chars={len(strategy)} destination={destination}")
PY
}

run_worker() {
  local dataset model gpu util sample min_acc min_syn max_steps output last_attempt total_cap
  local split_file split_name smiles_class smiles_task source_log seed log_file remaining offset
  local active_pid='' lookup_status lock_status dry_seed

  case "$CELL" in
    gsm-qwen35-2b)
      dataset=gsm_symbolic; model=Qwen/Qwen3.5-2B; gpu=0; util=0.40
      sample=49; min_acc=0.265306122449; min_syn=0.918367346939; max_steps=900
      last_attempt=10
      ;;
    gsm-qwen35-4b)
      dataset=gsm_symbolic; model=Qwen/Qwen3.5-4B; gpu=3; util=0.45
      sample=49; min_acc=0.448979591837; min_syn=0.918367346939; max_steps=900
      last_attempt=10
      ;;
    gsm-qwen35-9b)
      dataset=gsm_symbolic; model=Qwen/Qwen3.5-9B; gpu=2; util=0.60
      sample=49; min_acc=0.551020408163; min_syn=0.918367346939; max_steps=900
      last_attempt=3
      ;;
    smiles-qwen35-9b-isocyanates)
      dataset=smiles; model=Qwen/Qwen3.5-9B; gpu=2; util=0.60
      sample=50; min_acc=0.41; min_syn=0.90; max_steps=400
      last_attempt=1
      smiles_class=isocyanates
      smiles_task='Generate one new, valid, non-exemplar SMILES molecule for the isocyanates class. The answer contract is a single SMILES string and nothing else.'
      ;;
    spider-qwen35-4b)
      dataset=spider; model=Qwen/Qwen3.5-4B; gpu=3; util=0.45
      sample=300; min_acc=0.663333333333; min_syn=0.90; max_steps=900
      last_attempt=39
      ;;
    spider-qwen35-9b)
      dataset=spider; model=Qwen/Qwen3.5-9B; gpu=1; util=0.60
      sample=300; min_acc=0.673333333333; min_syn=0.90; max_steps=900
      last_attempt=37
      ;;
    spider-qwen25-7b)
      dataset=spider; model=Qwen/Qwen2.5-7B-Instruct; gpu=0; util=0.45
      sample=300; min_acc=0.66; min_syn=0.90; max_steps=900
      last_attempt=7
      ;;
    gsm14b)
      dataset=gsm_symbolic; model=Qwen/Qwen2.5-14B-Instruct; gpu=2; util=0.81
      sample=49; min_acc=0.5918; min_syn=0.85; max_steps=900
      last_attempt=33
      ;;
    *) log "unknown cell=$CELL"; return 2 ;;
  esac

  output=$(output_name_for_cell "$CELL") || {
    log "missing output mapping cell=$CELL"
    return 2
  }
  last_attempt="${RESUME_LAST_ATTEMPT:-$last_attempt}"
  total_cap="${RESUME_TOTAL_CAP:-40}"
  gpu="${RESUME_GPU:-$gpu}"
  if [[ ! "$total_cap" =~ ^[0-9]+$ ]] || (( total_cap < 2 || total_cap > 80 )); then
    log "invalid RESUME_TOTAL_CAP=$total_cap cell=$CELL"
    return 2
  fi
  if [[ ! "$last_attempt" =~ ^[0-9]+$ ]] || (( last_attempt < 1 || last_attempt >= total_cap )); then
    log "invalid RESUME_LAST_ATTEMPT=$last_attempt cell=$CELL"
    return 2
  fi

  source_log="${RESUME_SOURCE_LOG:-$REPO/outputs/generated/$output/run.log}"
  seed="${RESUME_SEED_FILE:-$RECOVERY_DIR/${CELL}_attempt${last_attempt}.dfy}"
  log_file="${RESUME_LOG_FILE:-$LOG_DIR/paid_synth_http429_resume_${CELL}.log}"

  if [[ "${DRY_RUN:-0}" == 1 ]]; then
    lookup_active_worker active_pid "$CELL"
    lookup_status=$?
    if (( lookup_status == 0 )); then
      log "DRY_RUN_REUSE cell=$CELL pid=$active_pid output=$output"
      return 0
    elif (( lookup_status != 1 )); then
      return "$lookup_status"
    fi
    if [[ -n "${RESUME_SEED_FILE:-}" ]]; then
      if [[ ! -s "$seed" ]]; then
        log "DRY_RUN_SEED_MISSING cell=$CELL seed=$seed"
        return 2
      fi
      log "DRY_RUN_SEED_OK cell=$CELL seed=$seed chars=$(wc -c <"$seed" | tr -d ' ')"
    else
      dry_seed=$(mktemp "${TMPDIR:-/tmp}/http429-${CELL}.XXXXXX.dfy") || return 2
      if ! extract_last_strategy "$source_log" "$last_attempt" "$dry_seed"; then
        rm -f "$dry_seed"
        return 2
      fi
      rm -f "$dry_seed"
    fi
    log "DRY_RUN_OK cell=$CELL replay_attempt=$last_attempt"
    return 0
  fi

  mkdir -p "$LOG_DIR" "$RECOVERY_DIR" "$LOCK_DIR"
  acquire_cell_lock
  lock_status=$?
  (( lock_status == 0 )) || return "$lock_status"
  lookup_active_worker active_pid "$CELL"
  lookup_status=$?
  if (( lookup_status == 0 )); then
    log "SKIP_DUPLICATE_ACTIVE cell=$CELL pid=$active_pid output=$output"
    return 75
  elif (( lookup_status != 1 )); then
    return "$lookup_status"
  fi
  if [[ -n "${RESUME_SEED_FILE:-}" ]]; then
    if [[ ! -s "$seed" ]]; then
      log "seed missing cell=$CELL seed=$seed"
      return 2
    fi
  else
    extract_last_strategy "$source_log" "$last_attempt" "$seed" || return $?
  fi

  remaining=$((total_cap - last_attempt + 1))
  offset=$((last_attempt - 1))
  split_file="$REPO/environment/benchmark_splits/gsm_symbolic_crane_proportional_49x49_seed123.json"
  split_name=train

  export LD_LIBRARY_PATH=/apps/conda/aadivyar/envs/csd/lib:${LD_LIBRARY_PATH:-}
  export CUDA_VISIBLE_DEVICES="$gpu"
  export HF_HOME=/home/aadivyar/.cache/huggingface
  export TRANSFORMERS_CACHE=/home/aadivyar/.cache/huggingface
  nohup "$PY" "$REPO/scripts/runtime/vllm_orphan_reaper.py" 9>&- \
    >>"$LOG_DIR/vllm_orphan_reaper.log" 2>&1 </dev/null &
  export CSD_DAILY_QUOTA_RETRY_SECONDS=3600
  export CSD_DAILY_QUOTA_RETRY_JITTER_SECONDS=300
  set -a; source "$REPO/.env"; set +a
  cd "$REPO"

  args=(
    --generation-model us.anthropic.claude-sonnet-4-6 --generation-backend bedrock
    --eval-model "$model" --eval-backend vllm
    --max-iterations "$remaining" --initial-attempt-offset "$offset"
    --initial-strategy-file "$seed"
    --output-name "$output" --output-dir "outputs/generated/$output"
    --min-accuracy "$min_acc" --min-syntax-rate "$min_syn"
    --eval-sample-size "$sample" --eval-max-steps "$max_steps" --eval-step-token-budget 1
    --eval-max-seconds-per-example 600 --eval-min-examples-before-threshold-stop "$sample"
    --max-tokens 32768 --restart-after-stuck-iters 0
    --vllm-gpu-memory-utilization "$util" --vllm-max-model-len 16384 --device auto
    --adaptive-helper-mask --helper-selection-policy bandit --refinement-beam-size 2
    --anthropic-thinking enabled --anthropic-effort high --anthropic-thinking-display summarized
    --vllm-tensor-parallel-size 1
  )
  if [[ -n "${RESUME_HISTORY_FILE:-}" ]]; then
    args+=(--initial-attempt-history-file "$RESUME_HISTORY_FILE")
  fi
  if [[ "$dataset" == gsm_symbolic ]]; then
    args+=(--dataset gsm_symbolic --gsm-split-file "$split_file" --gsm-split-name "$split_name")
    task='Solve math word problems step by step, wrapping intermediate symbolic expressions and the final answer inside << >> delimiters.'
  elif [[ "$dataset" == smiles ]]; then
    args+=(--dataset smiles --smiles-classes "$smiles_class" --smiles-samples-per-class "$sample")
    task="$smiles_task"
  else
    args+=(--dataset spider --spider-split-file environment/benchmark_splits/spider_dev_proportional_300x300_seed334.json --spider-split-name train)
    task='Generate a single valid SQL query as exactly `SQL: <<YOUR QUERY>>`, using only the provided schema context.'
  fi

  log "START cell=$CELL gpu=$gpu replay_attempt=$last_attempt next_new_attempt=$((last_attempt + 1)) total_cap=$total_cap log=$log_file"
  "$PY" -m synthesis.run_synthesis --task "$task" "${args[@]}" 2>&1 | tee -a "$log_file"
  status=${PIPESTATUS[0]}
  log "DONE cell=$CELL status=$status"
  return "$status"
}

wait_for_gpu2_then_launch_smiles() {
  local watched_pid="$1" watched_cell="$2" preverified_active="${3:-0}"
  local used active_pid='' lookup_status lock_status
  local active_seen="$preverified_active"

  if [[ "${DRY_RUN:-0}" == 1 ]]; then
    lookup_active_worker active_pid "$watched_cell"
    lookup_status=$?
    if (( lookup_status == 0 )); then
      if [[ "$active_pid" != "$watched_pid" ]]; then
        log "WAITER_REATTACH cell=$watched_cell old_pid=$watched_pid active_pid=$active_pid"
      fi
      log "DRY_RUN_WAITER_ACTIVE cell=$watched_cell pid=$active_pid"
      return 0
    elif (( lookup_status == 1 )); then
      log "DRY_RUN_WAITER_COMPLETE cell=$watched_cell last_pid=$watched_pid"
      if (( active_seen == 1 )); then
        log "DRY_RUN_WOULD_LAUNCH cell=smiles-qwen35-9b-isocyanates"
      fi
      return 0
    fi
    return "$lookup_status"
  fi

  acquire_waiter_lock
  lock_status=$?
  (( lock_status == 0 )) || return "$lock_status"
  log "WAIT cell=smiles-qwen35-9b-isocyanates after_cell=$watched_cell after_pid=$watched_pid gpu=2"
  while true; do
    active_pid=''
    lookup_active_worker active_pid "$watched_cell"
    lookup_status=$?
    if (( lookup_status == 0 )); then
      active_seen=1
      if [[ "$active_pid" != "$watched_pid" ]]; then
        log "WAITER_REATTACH cell=$watched_cell old_pid=$watched_pid active_pid=$active_pid"
        watched_pid="$active_pid"
      fi
      sleep "$POLL_SECONDS"
    elif (( lookup_status == 1 )); then
      if (( active_seen == 1 )); then
        break
      elif kill -0 "$watched_pid" 2>/dev/null; then
        sleep "$POLL_SECONDS"
      else
        log "WAITER_SOURCE_EXITED_BEFORE_ACTIVE cell=$watched_cell pid=$watched_pid"
        return 75
      fi
    else
      return "$lookup_status"
    fi
  done
  while true; do
    used=$(nvidia-smi --id=2 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    [[ "$used" =~ ^[0-9]+$ ]] && (( used < 1000 )) && break
    sleep "$POLL_SECONDS"
  done

  active_pid=''
  lookup_active_worker active_pid smiles-qwen35-9b-isocyanates
  lookup_status=$?
  if (( lookup_status == 0 )); then
    log "REUSE cell=smiles-qwen35-9b-isocyanates pid=$active_pid reason=already_running"
    return 0
  elif (( lookup_status != 1 )); then
    return "$lookup_status"
  fi
  nohup "$0" worker smiles-qwen35-9b-isocyanates 7>&- 8>&- >>"$QUEUE_LOG" 2>&1 &
  log "LAUNCHED cell=smiles-qwen35-9b-isocyanates pid=$! gpu=2"
}

case "$MODE" in
  worker) run_worker ;;
  controller)
    log "CONTROLLER_START override_scope=four_http429_interrupted_cells"
    p2=; p4=; p9=; pw=; planned2=; planned4=; planned9=; controller_status=
    gsm9_preverified=0
    if [[ "${DRY_RUN:-0}" != 1 ]]; then
      mkdir -p "$LOG_DIR" "$LOCK_DIR"
      acquire_controller_lock
      controller_status=$?
      (( controller_status == 0 )) || exit "$controller_status"
    fi
    plan_worker planned2 gsm-qwen35-2b || exit $?
    plan_worker planned4 gsm-qwen35-4b || exit $?
    plan_worker planned9 gsm-qwen35-9b || exit $?
    launch_planned_worker p2 gsm-qwen35-2b "$planned2"
    launch_planned_worker p4 gsm-qwen35-4b "$planned4"
    launch_planned_worker p9 gsm-qwen35-9b "$planned9"
    [[ -n "$planned9" ]] && gsm9_preverified=1
    if [[ "${DRY_RUN:-0}" == 1 ]]; then
      pw=none
      log "DRY_RUN_SKIP_WAITER gsm9_pid=$p9"
    else
      nohup "$0" waiter "$p9" gsm-qwen35-9b "$gsm9_preverified" 7>&- >>"$QUEUE_LOG" 2>&1 &
      pw=$!
    fi
    log "CONTROLLER_READY gsm2_pid=$p2 gsm4_pid=$p4 gsm9_pid=$p9 smiles_waiter_pid=$pw"
    ;;
  waiter) wait_for_gpu2_then_launch_smiles "$CELL" "${3:-gsm-qwen35-9b}" "${4:-0}" ;;
  *) log "unknown mode=$MODE"; exit 2 ;;
esac
