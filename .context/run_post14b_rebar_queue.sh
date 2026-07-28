#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/apps/conda/aadivyar/envs/csd/bin/python}"
MATRIX="${MATRIX:-$ROOT/.context/post14b_results_matrix.md}"
SNAPSHOT="${SNAPSHOT:-$ROOT/.context/post14b_pre_rebar_snapshot.json}"
MANIFEST="${MANIFEST:-$ROOT/.context/post14b_rebar_jobs.tsv}"
AUDIT="${AUDIT:-$ROOT/.context/post14b_rebar_audit.json}"
APPROVAL="${APPROVAL:-$ROOT/.context/post14b_rebar_approval.json}"
STATE="${STATE:-$ROOT/.context/post14b_rebar_state.tsv}"
CLAIMS_DIR="${CLAIMS_DIR:-$ROOT/.context/post14b_rebar_claims}"
RUN_SYNTH="${RUN_SYNTH:-$ROOT/run_synth_cell.sh}"
NVIDIA_SMI="${NVIDIA_SMI:-nvidia-smi}"
GPU_WAIT_POLL_SECONDS="${GPU_WAIT_POLL_SECONDS:-300}"
DRIVER_LOG="${DRIVER_LOG:-$ROOT/logs/post14b_rebar_queue_driver.log}"

mkdir -p "$(dirname "$DRIVER_LOG")"
set +e
"$PY" "$ROOT/.context/run_post14b_rebar_queue.py" \
  --repo "$ROOT" \
  --matrix "$MATRIX" \
  --snapshot "$SNAPSHOT" \
  --manifest "$MANIFEST" \
  --audit "$AUDIT" \
  --approval "$APPROVAL" \
  --state "$STATE" \
  --claims-dir "$CLAIMS_DIR" \
  --run-synth "$RUN_SYNTH" \
  --python "$PY" \
  --nvidia-smi "$NVIDIA_SMI" \
  --gpu-wait-poll-seconds "$GPU_WAIT_POLL_SECONDS" \
  "$@" 2>&1 | tee -a "$DRIVER_LOG"
exit_code=${PIPESTATUS[0]}
set -e
exit "$exit_code"
