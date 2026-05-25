#!/usr/bin/env bash
# Launch or attach a tmux session with the project conda env and .env loaded.
#
# Usage:
#   ./run_tmux.sh shell                 # interactive shell (default)
#   ./run_tmux.sh -d matrix -- [args]   # start detached (no attach)
#   ./run_tmux.sh run -- <command...>   # run command in tmux (logs to logs/tmux/)
#   ./run_tmux.sh matrix [-- args]      # run_all_tests.py (baselines + ablations; no metadecode by default)
#   ./run_tmux.sh baselines [-- args]   # same default strategies; often used with --skip-ablations
#   ./run_tmux.sh metadecode [-- args]  # run_all_tests.py with metadecode + synthesis ablations
#   ./run_tmux.sh synthesis [-- args]   # python -m synthesis.run_synthesis ...
#   ./run_tmux.sh attach [session]      # attach to existing session
#   ./run_tmux.sh kill [session]        # kill session
#
# Options (any position before subcommand args):
#   -d, --detached    create window/session but do not attach
#   -f, --fresh       kill existing session before starting (one window)
#
# Environment:
#   METADECODE_TMUX_SESSION     session name (default: metadecode)
#   METADECODE_CONDA_ENV        conda prefix (default: /apps/conda/advayth2/envs/advayth2)
#   CUDA_VISIBLE_DEVICES GPU for local runs (default: 2; set VAS_MAX_CUDA_DEVICES>1 to allow more)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="${METADECODE_TMUX_SESSION:-metadecode}"
DETACHED=0
FRESH=0
CONDA_ENV="${METADECODE_CONDA_ENV:-${METADECODE_RDKIT_CONDA_ENV:-/apps/conda/advayth2/envs/advayth2}}"
PYTHON="${CONDA_ENV}/bin/python"
# Matches run_all_tests.py DEFAULT_BASELINE_STRATEGIES (no metadecode).
LEGACY_BASELINE_STRATEGIES="${LEGACY_BASELINE_STRATEGIES:-unconstrained,gcd,crane,itergen,rejection_sampling}"
METADECODE_STRATEGIES="${METADECODE_STRATEGIES:-unconstrained,gcd,crane,itergen,rejection_sampling,metadecode}"

argv_has_flag() {
  local flag="$1"
  shift
  while [[ $# -gt 0 ]]; do
    if [[ "$1" == "$flag" ]]; then
      return 0
    fi
    shift
  done
  return 1
}

usage() {
  sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

require_tmux() {
  if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is not installed or not on PATH." >&2
    exit 1
  fi
}

require_python_env() {
  if [[ ! -x "$PYTHON" ]]; then
    echo "conda python not found: $PYTHON" >&2
    echo "Set METADECODE_CONDA_ENV to your env prefix." >&2
    exit 1
  fi
}

# Shell snippet run inside tmux (bash -lc).
# Bake GPU env from the invoking shell: tmux often carries a stale
# CUDA_VISIBLE_DEVICES (e.g. 2,3) that would override launch-time values.
project_env_script() {
  local cuda_visible="${CUDA_VISIBLE_DEVICES:-0}"
  local vas_max_cuda="${VAS_MAX_CUDA_DEVICES:-1}"
  local vas_tp="${VAS_VLLM_TENSOR_PARALLEL_SIZE:-$vas_max_cuda}"
  local vas_gpu_mem="${VAS_VLLM_GPU_MEMORY_UTILIZATION:-0.80}"
  local vas_mp="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
  cat <<EOF
set -euo pipefail
cd "$ROOT_DIR"
if [[ -f synthesis/.env ]]; then
  set -a
  # shellcheck disable=SC1091
  source synthesis/.env
  set +a
fi
export CONDA_PREFIX="$CONDA_ENV"
export PATH="$CONDA_ENV/bin:\${PATH:-}"
if [[ -d "$CONDA_ENV/lib" ]]; then
  export LD_LIBRARY_PATH="$CONDA_ENV/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
fi
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="$cuda_visible"
export VAS_MAX_CUDA_DEVICES="$vas_max_cuda"
export VAS_VLLM_TENSOR_PARALLEL_SIZE="$vas_tp"
export VAS_VLLM_GPU_MEMORY_UTILIZATION="$vas_gpu_mem"
export VLLM_WORKER_MULTIPROC_METHOD="$vas_mp"
EOF
}

attach_session() {
  local name="${1:-$SESSION}"
  exec tmux attach -t "$name"
}

kill_session() {
  local name="${1:-$SESSION}"
  if tmux has-session -t "$name" 2>/dev/null; then
    tmux kill-session -t "$name"
    echo "Killed tmux session: $name"
  else
    echo "No tmux session: $name" >&2
    exit 1
  fi
}

start_session() {
  local window_cmd="$1"
  local log_hint="${2:-}"
  ensure_tmux_log_dir
  if [[ "$FRESH" -eq 1 ]] && tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux kill-session -t "$SESSION"
  fi
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux new-window -t "$SESSION" -c "$ROOT_DIR" bash -lc "$window_cmd"
  else
    tmux new-session -d -s "$SESSION" -c "$ROOT_DIR" bash -lc "$window_cmd"
  fi
  echo "tmux session: $SESSION"
  if [[ "$DETACHED" -eq 1 ]]; then
    echo "Detached. Attach with: $0 attach ${SESSION}"
    if [[ -n "$log_hint" ]]; then
      echo "Log: $log_hint"
    fi
    return 0
  fi
  attach_session "$SESSION"
}

run_shell() {
  local inner
  inner="$(project_env_script)"
  inner+=$'\nexec bash -l\n'
  start_session "$inner" ""
}

ensure_tmux_log_dir() {
  mkdir -p "$ROOT_DIR/logs/tmux"
}

run_command() {
  if [[ $# -eq 0 ]]; then
    echo "run_tmux.sh run: missing command (use: ./run_tmux.sh run -- <cmd> ...)" >&2
    exit 1
  fi
  require_python_env
  ensure_tmux_log_dir
  local stamp
  stamp="$(date +%Y%m%d_%H%M%S)"
  local log_file="$ROOT_DIR/logs/tmux/${SESSION}_${stamp}.log"
  local quoted_cmd=""
  local quoted_log=""
  printf -v quoted_cmd '%q ' "$@"
  printf -v quoted_log '%q' "$log_file"

  local inner
  inner="$(project_env_script)"
  inner+=$'\n'
  inner+="echo \"Logging to: $log_file\""
  inner+=$'\n'
  inner+="echo \"Command: $quoted_cmd\""
  inner+=$'\n'
  inner+='set +e'
  inner+=$'\n'
  inner+="$quoted_cmd 2>&1 | tee -a $quoted_log"
  inner+=$'\n'
  inner+='echo "Exit code: ${PIPESTATUS[0]}" | tee -a '"$quoted_log"
  inner+=$'\n'
  inner+='exec bash -l'

  start_session "$inner" "$log_file"
}

run_matrix() {
  require_python_env
  local -a extra=()
  if [[ "${1:-}" == "--" ]]; then
    extra=("${@:1}")
  else
    extra=("$@")
  fi
  if argv_has_flag --strategies "${extra[@]}"; then
    run_command "$PYTHON" "$ROOT_DIR/run_all_tests.py" "${extra[@]}"
  else
    run_command "$PYTHON" "$ROOT_DIR/run_all_tests.py" \
      --strategies "$LEGACY_BASELINE_STRATEGIES" \
      "${extra[@]}"
  fi
}

run_baselines() {
  require_python_env
  local -a extra=()
  if [[ "${1:-}" == "--" ]]; then
    extra=("${@:1}")
  else
    extra=("$@")
  fi
  if argv_has_flag --strategies "${extra[@]}"; then
    run_matrix "${extra[@]}"
  else
    run_matrix --strategies "$LEGACY_BASELINE_STRATEGIES" "${extra[@]}"
  fi
}

run_metadecode_matrix() {
  require_python_env
  local -a extra=()
  if [[ "${1:-}" == "--" ]]; then
    extra=("${@:1}")
  else
    extra=("$@")
  fi
  if argv_has_flag --strategies "${extra[@]}"; then
    run_matrix "${extra[@]}"
  else
    run_matrix --strategies "$METADECODE_STRATEGIES" "${extra[@]}"
  fi
}

run_synthesis() {
  require_python_env
  run_command "$PYTHON" -m synthesis.run_synthesis "$@"
}

main() {
  local -a remaining=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -d|--detached)
        DETACHED=1
        shift
        ;;
      -f|--fresh)
        FRESH=1
        shift
        ;;
      *)
        remaining+=("$1")
        shift
        ;;
    esac
  done

  local cmd="${remaining[0]:-shell}"
  local -a args=()
  if [[ ${#remaining[@]} -gt 1 ]]; then
    args=("${remaining[@]:1}")
  fi

  case "$cmd" in
    -h|--help|help)
      usage 0
      ;;
    attach)
      require_tmux
      attach_session "${args[0]:-$SESSION}"
      ;;
    kill)
      require_tmux
      kill_session "${args[0]:-$SESSION}"
      ;;
    shell|"")
      require_tmux
      run_shell
      ;;
    run)
      require_tmux
      if [[ "${args[0]:-}" == "--" ]]; then
        run_command "${args[@]:1}"
      else
        run_command "${args[@]}"
      fi
      ;;
    matrix)
      require_tmux
      if [[ "${args[0]:-}" == "--" ]]; then
        run_matrix "${args[@]:1}"
      else
        run_matrix "${args[@]}"
      fi
      ;;
    baselines|legacy-baselines)
      require_tmux
      if [[ "${args[0]:-}" == "--" ]]; then
        run_baselines "${args[@]:1}"
      else
        run_baselines "${args[@]}"
      fi
      ;;
    metadecode|metadecode-matrix)
      require_tmux
      if [[ "${args[0]:-}" == "--" ]]; then
        run_metadecode_matrix "${args[@]:1}"
      else
        run_metadecode_matrix "${args[@]}"
      fi
      ;;
    synthesis)
      require_tmux
      if [[ "${args[0]:-}" == "--" ]]; then
        run_synthesis "${args[@]:1}"
      else
        run_synthesis "${args[@]}"
      fi
      ;;
    *)
      echo "Unknown command: $cmd" >&2
      usage 1
      ;;
  esac
}

main "$@"
