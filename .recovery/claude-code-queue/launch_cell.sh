#!/usr/bin/env bash
set -euo pipefail

[[ "${1:-}" == "worker" ]] || { echo "expected: worker <cell>" >&2; exit 2; }
CELL="${2:?missing cell}"
REPO="${REPO:-/home/aadivyar/csd-generation}"
PY="${PY:-/apps/conda/aadivyar/envs/csd/bin/python}"
MANIFEST="${CLAUDE_RECOVERY_MANIFEST:-$REPO/saved-results/2026-07-15-claude-helper-recovery-manifest.json}"
CHECKPOINT_ROOT="${CLAUDE_RECOVERY_CHECKPOINT_ROOT:-$REPO/.context/claude_recovery_queue_0715}"

exec "$PY" "$REPO/scripts/runtime/claude_recovery/launch_queue_cell.py" \
  --repo "$REPO" \
  --manifest "$MANIFEST" \
  --cell "$CELL" \
  --gpu "${RESUME_GPU:?missing RESUME_GPU}" \
  --python "$PY" \
  --checkpoint-root "$CHECKPOINT_ROOT" \
  --claude-executable /home/aadivyar/.local/bin/claude \
  --claude-config-dir /home/aadivyar/.claude-csd-synthesis \
  --expected-account aadivya@fermi.ai
