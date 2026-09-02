#!/bin/bash
# Wait for the gsm_7B baseline cell to finish, derive the bar, run GSM-7B metaDecode synthesis
# on seed123 TRAIN (GPU 2), then re-score the accepted strategy on the held-out EVAL split.
set -uo pipefail
cd /home/aadivyar/csd-generation
export CUDA_VISIBLE_DEVICES=2
export LD_LIBRARY_PATH=/opt/anaconda/lib:${LD_LIBRARY_PATH:-}
LOG=outputs/controlled_comparison/gsm7b_baselines_rerun.log
GSPLIT=environment/benchmark_splits/gsm_symbolic_crane_proportional_49x49_seed123.json
OUT=outputs/controlled_comparison/gsm_7B

until grep -q DONE_GSM7B_BASELINES "$LOG" 2>/dev/null; do sleep 120; done
echo "BASELINES_DONE $(date)"

read MINACC <<<"$(python - <<'PY'
import json, glob
best = 0.0
for p in glob.glob("outputs/controlled_comparison/gsm_7B/*.json"):
    if p.endswith("metadecode.json"): continue
    try: d = json.load(open(p))
    except Exception: continue
    a = float(d.get("accuracy") or 0.0)
    if a > 0.05:  # ignore known-broken empties
        best = max(best, a)
print(f"{min(1.0, (int(round(best*49))+1)/49):.4f}")
PY
)"
echo "BAR_GSM7B: min-accuracy=$MINACC min-syntax=0.90"

NAME=gsm7b_seed123_fresh_20260610
python -m synthesis.run_synthesis \
  --task "Solve math word problems step by step, wrapping intermediate symbolic expressions and the final answer inside << >> delimiters." \
  --dataset gsm_symbolic \
  --generation-model us.anthropic.claude-sonnet-4-6 \
  --generation-backend bedrock \
  --anthropic-thinking enabled --anthropic-effort high \
  --anthropic-thinking-display summarized \
  --eval-model Qwen/Qwen2.5-7B-Instruct --eval-backend vllm \
  --max-iterations 20 \
  --output-name "$NAME" --output-dir "outputs/generated/$NAME" \
  --min-accuracy "$MINACC" --min-syntax-rate 0.90 \
  --eval-sample-size 49 --eval-max-steps 900 --eval-step-token-budget 1 \
  --eval-max-seconds-per-example 120 \
  --eval-min-examples-before-threshold-stop 49 \
  --max-tokens 32768 \
  --restart-after-stuck-iters 0 \
  --vllm-gpu-memory-utilization 0.50 --device auto --vllm-tensor-parallel-size 1 \
  --adaptive-helper-mask --helper-selection-policy bandit \
  --refinement-beam-size 2 \
  --gsm-split-file "$GSPLIT" --gsm-split-name train
echo "EXIT_SYNTH_GSM7B=$?"

CSD=$(ls -t outputs/generated/${NAME}*/python/GeneratedCSD.py 2>/dev/null | head -1)
if [ -n "$CSD" ]; then
  python -m synthesis.scripts.reevaluate_compiled_csd "$CSD" \
    --dataset gsm_symbolic --eval-model Qwen/Qwen2.5-7B-Instruct --eval-backend vllm \
    --sample-size 49 --max-steps 900 --step-token-budget 1 \
    --vllm-gpu-memory-utilization 0.50 \
    --gsm-split-file "$GSPLIT" --gsm-split-name eval \
    --output-json "$OUT/metadecode.json"
  echo "EXIT_REEVAL_GSM7B=$?"
else
  echo "NO_ACCEPTED_CSD_GSM7B"
fi
echo DONE_GSM7B_CHAIN
