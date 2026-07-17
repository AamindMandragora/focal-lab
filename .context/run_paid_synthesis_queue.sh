#!/usr/bin/env bash
set -u
set -o pipefail

REPO="${REPO:-/home/aadivyar/csd-generation}"
PY="${PY:-/apps/conda/aadivyar/envs/csd/bin/python}"
STATUS="${STATUS:-logs/paid_synthesis_queue_status.tsv}"
APPROVAL_NOTE="${APPROVAL_NOTE:-saved-results/2026-07-08-bedrock-paid-synthesis-approval.md}"
GPU="${GPU:-3}"
GPU_WAIT_MAX_USED_MIB="${GPU_WAIT_MAX_USED_MIB:-1000}"
DRY_RUN="${DRY_RUN:-0}"
ONLY_LABEL="${ONLY_LABEL:-}"
FINALIZE_ONLY="${FINALIZE_ONLY:-0}"
PAID_JOB_SET="${PAID_JOB_SET:-fixed}"
CLAIMS_DIR="${CLAIMS_DIR:-.context/paid_synthesis_fixed_claims}"

cd "$REPO"
mkdir -p logs "$CLAIMS_DIR" outputs/controlled_comparison/gsm_14B outputs/controlled_comparison/spider_14B

record_status() {
  local started_at="$1"
  local finished_at="$2"
  local label="$3"
  local status="$4"
  local exit_code="$5"
  local output_json="$6"
  local log_path="$7"

  {
    flock -x 9
    if [ ! -s "$STATUS" ]; then
      printf "started_at\tfinished_at\tlabel\tstatus\texit_code\toutput_json\tlog\n" >&9
    fi
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$started_at" "$finished_at" "$label" "$status" "$exit_code" "$output_json" "$log_path" >&9
  } 9>> "$STATUS"
}

latest_status() {
  local label="$1"
  if [ ! -s "$STATUS" ]; then
    return 0
  fi
  {
    flock -s 9
    awk -F '\t' -v wanted="$label" 'NR > 1 && $3 == wanted {status=$4} END {print status}' "$STATUS"
  } 9< "$STATUS"
}

is_terminal_status() {
  case "$1" in
    heldout_failed|heldout_gpu_failed|skip_exists|ran|no_success_csd|synthesis_failed|gpu_wait_failed|interrupted) return 0 ;;
    *) return 1 ;;
  esac
}

selected_label() {
  [ -z "$ONLY_LABEL" ] || [ "$ONLY_LABEL" = "$1" ]
}

resolve_job_artifacts() {
  local label="$1"
  local output_name="$2"
  local output_json="$3"
  local log_path="$4"
  JOB_OUTPUT_NAME="$output_name"
  JOB_OUTPUT_JSON="$output_json"
  JOB_LOG_PATH="$log_path"
  if [ "$PAID_JOB_SET" = "infra-retry" ]; then
    JOB_OUTPUT_NAME="${output_name}_infraretry_0711"
    JOB_OUTPUT_JSON="${output_json%.json}_infraretry_0711.json"
    JOB_LOG_PATH="logs/paid_synth_infra_retry_${label//-/_}.log"
  fi
}

require_paid_approval() {
  if [ ! -s "$APPROVAL_NOTE" ]; then
    echo "missing paid approval note: $APPROVAL_NOTE" >&2
    return 2
  fi
  if ! grep -q "User approval is explicit for paid Bedrock synthesis" "$APPROVAL_NOTE"; then
    echo "approval note exists but does not contain the required approval marker" >&2
    return 2
  fi
  if [ ! -s "$REPO/.env" ] || ! grep -q '^AWS_BEARER_TOKEN_BEDROCK=' "$REPO/.env"; then
    echo "missing AWS_BEARER_TOKEN_BEDROCK in $REPO/.env" >&2
    return 2
  fi
  if [ "$PAID_JOB_SET" = "infra-retry" ]; then
    "$PY" "$REPO/.context/run_paid_synthesis_pool.py" \
      --repo "$REPO" \
      --job-set "$PAID_JOB_SET" \
      --verify-worker-launch \
      --worker-status "$STATUS" \
      --worker-claims-dir "$CLAIMS_DIR" || return $?
  else
    "$PY" "$REPO/.context/run_paid_synthesis_pool.py" \
      --repo "$REPO" \
      --job-set "$PAID_JOB_SET" \
      --verify-only || return $?
  fi
}

wait_for_gpu() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi missing; cannot verify GPU availability" >&2
    return 2
  fi

  while true; do
    local used
    if ! used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')"; then
      echo "nvidia-smi failed while checking GPU $GPU" >&2
      return 2
    fi
    if [ -n "$used" ] && [ "$used" -le "$GPU_WAIT_MAX_USED_MIB" ]; then
      echo "[paid-synthesis-queue] GPU $GPU ready with ${used}MiB used"
      return 0
    fi
    echo "[paid-synthesis-queue] waiting for GPU $GPU: ${used:-unknown}MiB used at $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    sleep 300
  done
}

compiled_csd_for_run() {
  local output_name="$1"
  "$PY" - "$output_name" <<'PY'
import json
import sys
from pathlib import Path

name = sys.argv[1]
latest = Path("outputs/generated") / name / "latest_run.txt"
if not latest.exists():
    raise SystemExit(1)
run_dir = Path(latest.read_text().strip())
report = run_dir / "results" / "success_report.json"
if not report.exists():
    raise SystemExit(1)
compiled_dir = json.load(open(report)).get("compiled_dir", "")
if not compiled_dir:
    raise SystemExit(1)
csd = Path(compiled_dir) / "GeneratedCSD.py"
if not csd.exists():
    raise SystemExit(1)
print(csd)
PY
}

run_paid_job() {
  local label="$1"
  local output_name="$2"
  local output_json="$3"
  local log_path="$4"
  shift 4

  mkdir -p "$(dirname "$output_json")" "$(dirname "$log_path")"
  local started_at
  started_at="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

  local prior_status
  prior_status="$(latest_status "$label")"
  if is_terminal_status "$prior_status"; then
    echo "[paid-synthesis-queue] terminal state already recorded; never relaunching $label status=$prior_status"
    return 0
  fi

  if [ -s "$output_json" ]; then
    mkdir "$CLAIMS_DIR/$label" 2>/dev/null || true
    echo "[paid-synthesis-queue] skip existing $label -> $output_json"
    record_status "$started_at" "$started_at" "$label" "skip_exists" "0" "$output_json" "$log_path"
    return 0
  fi

  if ! mkdir "$CLAIMS_DIR/$label"; then
    echo "[paid-synthesis-queue] claim already exists; refusing a second paid cycle for $label" >&2
    return 3
  fi

  if [ "$DRY_RUN" = "1" ]; then
    echo "[paid-synthesis-queue] dry-run $label -> $output_json"
    printf "DRY_RUN %s\n" "$label" > "$log_path"
    record_status "$started_at" "$started_at" "$label" "dry_run" "0" "$output_json" "$log_path"
    return 0
  fi

  wait_for_gpu >> "$log_path" 2>&1 || {
    local exit_code=$?
    printf "heldout_precondition=gpu_wait_failed exit=%s\n" "$exit_code" >> "$log_path"
    record_status "$started_at" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$label" "gpu_wait_failed" "$exit_code" "$output_json" "$log_path"
    return "$exit_code"
  }

  echo "[paid-synthesis-queue] start $label -> $output_name"
  "$@" > "$log_path" 2>&1
  local synth_exit=$?

  if [ "$synth_exit" -eq 0 ]; then
    local csd
    if csd="$(compiled_csd_for_run "$output_name")"; then
      wait_for_gpu >> "$log_path" 2>&1 || {
        local gpu_exit=$?
        local status="heldout_gpu_failed"
        printf "heldout_precondition=gpu_wait_failed exit=%s\n" "$gpu_exit" >> "$log_path"
        record_status "$started_at" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$label" "$status" "$gpu_exit" "$output_json" "$log_path"
        echo "[paid-synthesis-queue] finish $label heldout GPU check failed exit=$gpu_exit"
        return "$gpu_exit"
      }
      reeval_compiled "$label" "$csd" "$output_json" >> "$log_path" 2>&1
      local reeval_exit=$?
      local reeval_status="ran"
      if [ "$reeval_exit" -ne 0 ] || [ ! -s "$output_json" ]; then
        reeval_status="heldout_failed"
        printf "heldout_result=failed exit=%s output_exists=%s\n" \
          "$reeval_exit" "$([ -s "$output_json" ] && echo yes || echo no)" >> "$log_path"
      fi
      record_status "$started_at" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$label" "$reeval_status" "$reeval_exit" "$output_json" "$log_path"
      echo "[paid-synthesis-queue] finish $label status=$reeval_status reeval_exit=$reeval_exit"
      return "$reeval_exit"
    fi
    record_status "$started_at" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$label" "no_success_csd" "$synth_exit" "$output_json" "$log_path"
    printf "synthesis_result=no_success_csd exit=%s\n" "$synth_exit" >> "$log_path"
    echo "[paid-synthesis-queue] finish $label no accepted CSD"
    return 0
  fi

  printf "synthesis_result=failed exit=%s\n" "$synth_exit" >> "$log_path"
  record_status "$started_at" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$label" "synthesis_failed" "$synth_exit" "$output_json" "$log_path"
  echo "[paid-synthesis-queue] finish $label synth_exit=$synth_exit"
  return "$synth_exit"
}

reeval_compiled() {
  local label="$1"
  local csd="$2"
  local output_json="$3"

  case "$label" in
    gsm14b)
      "$PY" -m synthesis.scripts.reevaluate_compiled_csd "$csd" \
        --dataset gsm_symbolic \
        --eval-model "Qwen/Qwen2.5-14B-Instruct" \
        --eval-backend vllm \
        --device auto \
        --sample-size 49 \
        --max-steps 900 \
        --step-token-budget 1 \
        --max-seconds-per-example 600 \
        --vllm-gpu-memory-utilization 0.80 \
        --vllm-tensor-parallel-size 1 \
        --gsm-split-file environment/benchmark_splits/gsm_symbolic_crane_proportional_49x49_seed123.json \
        --gsm-split-name eval \
        --output-json "$output_json"
      ;;
    spider14b)
      "$PY" -m synthesis.scripts.reevaluate_compiled_csd "$csd" \
        --dataset spider \
        --eval-model "Qwen/Qwen2.5-14B-Instruct" \
        --eval-backend vllm \
        --device auto \
        --sample-size 300 \
        --max-steps 900 \
        --step-token-budget 1 \
        --max-seconds-per-example 600 \
        --vllm-gpu-memory-utilization 0.80 \
        --vllm-tensor-parallel-size 1 \
        --spider-split-file environment/benchmark_splits/spider_dev_proportional_300x300_seed334.json \
        --spider-split-name test \
        --output-json "$output_json"
      ;;
    smiles-*)
      local class_name
      local smiles_tag
      local smiles_model
      local smiles_util
      class_name="$(basename "$(dirname "$output_json")")"
      smiles_tag="$(basename "$(dirname "$(dirname "$output_json")")")"
      case "$smiles_tag" in
        smiles_qwen35_2b)
          smiles_model="Qwen/Qwen3.5-2B"
          smiles_util=0.40
          ;;
        smiles_qwen35_4b)
          smiles_model="Qwen/Qwen3.5-4B"
          smiles_util=0.45
          ;;
        smiles_qwen35_9b)
          smiles_model="Qwen/Qwen3.5-9B"
          smiles_util=0.60
          ;;
        *)
          echo "unknown SMILES output tag: $smiles_tag" >&2
          return 2
          ;;
      esac
      "$PY" -m synthesis.scripts.reevaluate_compiled_csd "$csd" \
        --dataset smiles \
        --smiles-classes "$class_name" \
        --eval-model "$smiles_model" \
        --eval-backend vllm \
        --device auto \
        --sample-size 100 \
        --max-steps 400 \
        --step-token-budget 1 \
        --vllm-gpu-memory-utilization "$smiles_util" \
        --vllm-tensor-parallel-size 1 \
        --output-json "$output_json"
      ;;
    *)
      echo "unknown reeval label: $label" >&2
      return 2
      ;;
  esac
}

require_paid_approval || exit $?

case "$ONLY_LABEL" in
  ""|smiles-qwen35-2b-chain_extenders|smiles-qwen35-4b-acrylates|smiles-qwen35-4b-chain_extenders|smiles-qwen35-9b-acrylates|smiles-qwen35-9b-chain_extenders|gsm14b|spider14b) ;;
  *) echo "unknown ONLY_LABEL=$ONLY_LABEL" >&2; exit 2 ;;
esac

if [ "$FINALIZE_ONLY" != "1" ]; then
if selected_label "smiles-qwen35-2b-chain_extenders"; then
run_paid_job "smiles-qwen35-2b-chain_extenders" "synth_smiles_qwen35_2b_chain_extenders_retry_0708" \
  "outputs/controlled_comparison/smiles_qwen35_2b/chain_extenders/metadecode_uv_paid0708.json" \
  "logs/paid_synth_smiles_qwen35_2b_chain_extenders.log" \
  env DATASET=smiles EVAL_MODEL="Qwen/Qwen3.5-2B" GPU="$GPU" GPU_MEM_UTIL=0.40 ANTHROPIC_THINKING=enabled SPLIT_NAME=train SPLIT_FILE= \
  MAX_ITERS=40 SAMPLE_SIZE=50 MIN_ACC=0.42 MIN_SYN=0.90 EVAL_MAX_STEPS=400 \
  OUTPUT_NAME="synth_smiles_qwen35_2b_chain_extenders_retry_0708" \
  SMILES_CLASS=chain_extenders \
  SMILES_TASK="Generate one new, valid, non-exemplar SMILES molecule for the chain_extenders class. The answer contract is a single SMILES string and nothing else." \
  bash run_synth_cell.sh
fi

if selected_label "smiles-qwen35-4b-acrylates"; then
resolve_job_artifacts "smiles-qwen35-4b-acrylates" "synth_smiles_qwen35_4b_acrylates_retry_0708" \
  "outputs/controlled_comparison/smiles_qwen35_4b/acrylates/metadecode_uv_paid0708.json" "logs/paid_synth_smiles_qwen35_4b_acrylates.log"
run_paid_job "smiles-qwen35-4b-acrylates" "$JOB_OUTPUT_NAME" "$JOB_OUTPUT_JSON" "$JOB_LOG_PATH" \
  env DATASET=smiles EVAL_MODEL="Qwen/Qwen3.5-4B" GPU="$GPU" GPU_MEM_UTIL=0.45 ANTHROPIC_THINKING=enabled SPLIT_NAME=train SPLIT_FILE= \
  MAX_ITERS=40 SAMPLE_SIZE=50 MIN_ACC=0.38 MIN_SYN=0.90 EVAL_MAX_STEPS=400 \
  OUTPUT_NAME="$JOB_OUTPUT_NAME" \
  SMILES_CLASS=acrylates \
  SMILES_TASK="Generate one new, valid, non-exemplar SMILES molecule for the acrylates class. The answer contract is a single SMILES string and nothing else." \
  bash run_synth_cell.sh
fi

if selected_label "smiles-qwen35-4b-chain_extenders"; then
resolve_job_artifacts "smiles-qwen35-4b-chain_extenders" "synth_smiles_qwen35_4b_chain_extenders_retry_0708" \
  "outputs/controlled_comparison/smiles_qwen35_4b/chain_extenders/metadecode_uv_paid0708.json" "logs/paid_synth_smiles_qwen35_4b_chain_extenders.log"
run_paid_job "smiles-qwen35-4b-chain_extenders" "$JOB_OUTPUT_NAME" "$JOB_OUTPUT_JSON" "$JOB_LOG_PATH" \
  env DATASET=smiles EVAL_MODEL="Qwen/Qwen3.5-4B" GPU="$GPU" GPU_MEM_UTIL=0.45 ANTHROPIC_THINKING=enabled SPLIT_NAME=train SPLIT_FILE= \
  MAX_ITERS=40 SAMPLE_SIZE=50 MIN_ACC=0.62 MIN_SYN=0.90 EVAL_MAX_STEPS=400 \
  OUTPUT_NAME="$JOB_OUTPUT_NAME" \
  SMILES_CLASS=chain_extenders \
  SMILES_TASK="Generate one new, valid, non-exemplar SMILES molecule for the chain_extenders class. The answer contract is a single SMILES string and nothing else." \
  bash run_synth_cell.sh
fi

if selected_label "smiles-qwen35-9b-acrylates"; then
run_paid_job "smiles-qwen35-9b-acrylates" "synth_smiles_qwen35_9b_acrylates_retry_0708" \
  "outputs/controlled_comparison/smiles_qwen35_9b/acrylates/metadecode_uv_paid0708.json" \
  "logs/paid_synth_smiles_qwen35_9b_acrylates.log" \
  env DATASET=smiles EVAL_MODEL="Qwen/Qwen3.5-9B" GPU="$GPU" GPU_MEM_UTIL=0.60 ANTHROPIC_THINKING=enabled SPLIT_NAME=train SPLIT_FILE= \
  MAX_ITERS=40 SAMPLE_SIZE=50 MIN_ACC=0.34 MIN_SYN=0.90 EVAL_MAX_STEPS=400 \
  OUTPUT_NAME="synth_smiles_qwen35_9b_acrylates_retry_0708" \
  SMILES_CLASS=acrylates \
  SMILES_TASK="Generate one new, valid, non-exemplar SMILES molecule for the acrylates class. The answer contract is a single SMILES string and nothing else." \
  bash run_synth_cell.sh
fi

if selected_label "smiles-qwen35-9b-chain_extenders"; then
run_paid_job "smiles-qwen35-9b-chain_extenders" "synth_smiles_qwen35_9b_chain_extenders_retry_0708" \
  "outputs/controlled_comparison/smiles_qwen35_9b/chain_extenders/metadecode_uv_paid0708.json" \
  "logs/paid_synth_smiles_qwen35_9b_chain_extenders.log" \
  env DATASET=smiles EVAL_MODEL="Qwen/Qwen3.5-9B" GPU="$GPU" GPU_MEM_UTIL=0.60 ANTHROPIC_THINKING=enabled SPLIT_NAME=train SPLIT_FILE= \
  MAX_ITERS=40 SAMPLE_SIZE=50 MIN_ACC=0.58 MIN_SYN=0.90 EVAL_MAX_STEPS=400 \
  OUTPUT_NAME="synth_smiles_qwen35_9b_chain_extenders_retry_0708" \
  SMILES_CLASS=chain_extenders \
  SMILES_TASK="Generate one new, valid, non-exemplar SMILES molecule for the chain_extenders class. The answer contract is a single SMILES string and nothing else." \
  bash run_synth_cell.sh
fi

if selected_label "gsm14b"; then
resolve_job_artifacts "gsm14b" "synth_gsm14b_z3bar_retry_0708" \
  "outputs/controlled_comparison/gsm_14B/metadecode_paid0708.json" "logs/paid_synth_gsm14b_z3bar.log"
run_paid_job "gsm14b" "$JOB_OUTPUT_NAME" "$JOB_OUTPUT_JSON" "$JOB_LOG_PATH" \
  env DATASET=gsm_symbolic EVAL_MODEL="Qwen/Qwen2.5-14B-Instruct" GPU="$GPU" GPU_MEM_UTIL=0.80 ANTHROPIC_THINKING=enabled SPLIT_NAME=train SPLIT_FILE=/home/aadivyar/csd-generation/environment/benchmark_splits/gsm_symbolic_crane_proportional_49x49_seed123.json \
  MAX_ITERS=40 SAMPLE_SIZE=49 MIN_ACC=0.5918 MIN_SYN=0.85 EVAL_MAX_STEPS=900 EVAL_MAX_SECONDS=600 \
  OUTPUT_NAME="$JOB_OUTPUT_NAME" SPLIT_NAME=train \
  bash run_synth_cell.sh
fi

if selected_label "spider14b"; then
resolve_job_artifacts "spider14b" "synth_spider14b_retry_0708" \
  "outputs/controlled_comparison/spider_14B/metadecode_paid0708.json" "logs/paid_synth_spider14b.log"
run_paid_job "spider14b" "$JOB_OUTPUT_NAME" "$JOB_OUTPUT_JSON" "$JOB_LOG_PATH" \
  env DATASET=spider EVAL_MODEL="Qwen/Qwen2.5-14B-Instruct" GPU="$GPU" GPU_MEM_UTIL=0.80 ANTHROPIC_THINKING=enabled SPLIT_NAME=train SPLIT_FILE=/home/aadivyar/csd-generation/environment/benchmark_splits/spider_dev_proportional_300x300_seed334.json \
  MAX_ITERS=40 SAMPLE_SIZE=300 MIN_ACC=0.647 MIN_SYN=0.85 EVAL_MAX_STEPS=900 EVAL_MAX_SECONDS=600 \
  OUTPUT_NAME="$JOB_OUTPUT_NAME" SPLIT_NAME=train \
  bash run_synth_cell.sh
fi
fi

if [ -n "$ONLY_LABEL" ]; then
  exit 0
fi

FIXED_COMPLETE="${FIXED_COMPLETE:-logs/paid_synthesis_fixed_complete.json}"
if [ "$DRY_RUN" != "1" ]; then
  "$PY" - "$STATUS" "$FIXED_COMPLETE" "$PAID_JOB_SET" <<'PY'
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

status_path = Path(sys.argv[1])
marker_path = Path(sys.argv[2])
job_set = sys.argv[3]
fixed_required = [
    "smiles-qwen35-2b-chain_extenders",
    "smiles-qwen35-4b-acrylates",
    "smiles-qwen35-4b-chain_extenders",
    "smiles-qwen35-9b-acrylates",
    "smiles-qwen35-9b-chain_extenders",
    "gsm14b",
    "spider14b",
]
infra_retry_required = [
    "smiles-qwen35-4b-acrylates",
    "smiles-qwen35-4b-chain_extenders",
    "gsm14b",
    "spider14b",
]
required = fixed_required if job_set == "fixed" else infra_retry_required
terminal = {
    "heldout_failed",
    "heldout_gpu_failed",
    "skip_exists",
    "ran",
    "no_success_csd",
    "synthesis_failed",
    "gpu_wait_failed",
}
latest = {}
with status_path.open(newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle, delimiter="\t"):
        if row.get("label") in required:
            latest[row["label"]] = row
missing = [label for label in required if label not in latest]
nonterminal = [
    label for label in required
    if label in latest and latest[label].get("status") not in terminal
]
result_statuses = {"ran", "skip_exists"}
failure_statuses = terminal - result_statuses
field_errors = []
failure_artifact_errors = []
result_artifact_errors = []
for label in required:
    row = latest.get(label)
    if row is None:
        continue
    if not row.get("finished_at") or not row.get("exit_code"):
        field_errors.append(label)
    if row.get("status") in failure_statuses:
        log_path = Path(row.get("log", ""))
        if not log_path.is_file() or log_path.stat().st_size == 0:
            failure_artifact_errors.append(f"{label}:{log_path}")
    if row.get("status") in result_statuses:
        output_path = Path(row.get("output_json", ""))
        if not output_path.is_file() or output_path.stat().st_size == 0:
            result_artifact_errors.append(f"{label}:{output_path}")
if missing or nonterminal or field_errors or failure_artifact_errors or result_artifact_errors:
    raise SystemExit(
        "fixed paid queue is not complete: "
        f"missing={missing}, nonterminal={nonterminal}, fields={field_errors}, "
        f"failure_artifacts={failure_artifact_errors}, result_artifacts={result_artifact_errors}"
    )
payload = {
    "status": "complete",
    "completed_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    "required_labels": required,
    "jobs": {label: latest[label] for label in required},
}
marker_path.parent.mkdir(parents=True, exist_ok=True)
temporary = marker_path.with_suffix(marker_path.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, marker_path)
print(f"[paid-synthesis-queue] fixed queue terminal marker: {marker_path}")
PY
fi
