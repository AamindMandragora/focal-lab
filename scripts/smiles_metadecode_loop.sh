#!/usr/bin/env bash
# Continuous SMILES metadecode synthesis loop for run_all_tests.py.
# Rotates step budgets and iteration counts; logs each pass under logs/tmux/.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${CONDA_PREFIX:-/apps/conda/advayth2/envs/advayth2}/bin/python"
LOG_DIR="$ROOT_DIR/logs/tmux"
mkdir -p "$LOG_DIR"

cd "$ROOT_DIR"
if [[ -f synthesis/.env ]]; then
  set -a
  # shellcheck disable=SC1091
  source synthesis/.env
  set +a
fi

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
export VAS_MAX_CUDA_DEVICES="${VAS_MAX_CUDA_DEVICES:-2}"
export VAS_VLLM_TENSOR_PARALLEL_SIZE="${VAS_VLLM_TENSOR_PARALLEL_SIZE:-2}"
export VAS_VLLM_GPU_MEMORY_UTILIZATION="${VAS_VLLM_GPU_MEMORY_UTILIZATION:-0.80}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VAS_SMILES_EVAL_MAX_SECONDS_PER_EXAMPLE="${VAS_SMILES_EVAL_MAX_SECONDS_PER_EXAMPLE:-600}"

# Prefer ms256 first: CARS targets already complete for all three classes.
STEP_BUDGETS=(256 512 900)
SYNTH_ITERS=(10 15)
pass=0

echo "[smiles_metadecode_loop] starting continuous metadecode SMILES runs"
echo "[smiles_metadecode_loop] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[smiles_metadecode_loop] target: beat CARS ~29% mean accuracy (7B, tb1)"

while true; do
  pass=$((pass + 1))
  ms="${STEP_BUDGETS[$((pass % ${#STEP_BUDGETS[@]}))]}"
  iters="${SYNTH_ITERS[$((pass % ${#SYNTH_ITERS[@]}))]}"
  stamp="$(date +%Y%m%d_%H%M%S)"
  log_file="$LOG_DIR/smiles_metadecode_pass${pass}_${stamp}.log"

  echo ""
  echo "=== pass $pass @ $(date -Is) ms=$ms iters=$iters ===" | tee -a "$log_file"

  set +e
  "$PYTHON" "$ROOT_DIR/run_all_tests.py" \
    --benchmarks smiles \
    --strategies metadecode \
    --models "Qwen/Qwen2.5-Coder-7B-Instruct" \
    --generation-models gpt5.4 \
    --skip-ablations \
    --eval-max-steps "$ms" \
    --synthesis-iterations "$iters" \
    --eval-min-examples-before-threshold-stop 50 \
    2>&1 | tee -a "$log_file"
  rc=${PIPESTATUS[0]}
  set -e

  echo "=== pass $pass exit=$rc @ $(date -Is) ===" | tee -a "$log_file"

  # Report best metadecode result so far for this step budget.
  "$PYTHON" - <<'PY' "$ROOT_DIR" "$ms" | tee -a "$log_file"
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
ms = sys.argv[2]
cars_root = root / "outputs/baselines/cars/Qwen_Qwen2.5_Coder_7B_Instruct"
cars_accs = []
for p in cars_root.glob(f"smiles__*__tb1__ms{ms}__*.json"):
    try:
        d = json.loads(p.read_text())
        cars_accs.append(float(d.get("accuracy", 0)))
    except Exception:
        pass
cars_mean = sum(cars_accs) / len(cars_accs) if cars_accs else 0.0

best = None
for p in (root / "outputs/baselines/metadecode/Qwen_Qwen2.5_Coder_7B_Instruct").glob(
    f"smiles__tb1__ms{ms}__*.json"
):
    try:
        d = json.loads(p.read_text())
        acc = float(d.get("accuracy", 0))
        syn = float(d.get("syntax_rate", 0))
        if best is None or acc > best[0]:
            best = (acc, syn, p.name)
    except Exception:
        pass

print(f"[scoreboard] CARS mean acc ms{ms}: {cars_mean:.1%}")
if best:
    print(f"[scoreboard] best metadecode ms{ms}: acc={best[0]:.1%} syn={best[1]:.1%} ({best[2]})")
    if best[0] > cars_mean:
        print(f"[scoreboard] BEATING CARS on ms{ms} accuracy")
else:
    print(f"[scoreboard] no metadecode JSON yet for ms{ms}")
PY

  sleep 10
done
