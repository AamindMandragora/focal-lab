#!/bin/bash
# Wait for GSM-1.5B synthesis to finish, then re-score the accepted strategy on the
# held-out seed123 EVAL split (GPU 0) -> outputs/controlled_comparison/gsm_1p5B/metadecode.json
set -uo pipefail
cd /home/aadivyar/csd-generation
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/opt/anaconda/lib:${LD_LIBRARY_PATH:-}
LOG=outputs/gsm1p5b_seed123_fresh_20260610.log
GSPLIT=environment/benchmark_splits/gsm_symbolic_crane_proportional_49x49_seed123.json
NAME=gsm1p5b_seed123_fresh_20260610

until grep -q DONE_GSM1P5B_SYNTH "$LOG" 2>/dev/null; do sleep 120; done
echo "SYNTH_DONE $(date)"

# Accepted-strategy pickup: rejected attempts also leave GeneratedCSD.py files, so a
# bare glob is wrong (and misses the nested run dir anyway). The reliable pointer is
# latest_run.txt -> results/success_report.json -> compiled_dir.
RUNDIR=$(cat "outputs/generated/$NAME/latest_run.txt" 2>/dev/null)
CSD=""
if [ -n "$RUNDIR" ] && [ -s "$RUNDIR/results/success_report.json" ]; then
  CDIR=$(python -c "import json,sys;print(json.load(open(sys.argv[1])).get('compiled_dir',''))" "$RUNDIR/results/success_report.json" 2>/dev/null)
  [ -n "$CDIR" ] && [ -s "$CDIR/GeneratedCSD.py" ] && CSD="$CDIR/GeneratedCSD.py"
fi

if [ -n "$CSD" ]; then
  echo "ACCEPTED_CSD=$CSD"
  python -m synthesis.scripts.reevaluate_compiled_csd "$CSD" \
    --dataset gsm_symbolic --eval-model Qwen/Qwen2.5-1.5B-Instruct --eval-backend vllm \
    --sample-size 49 --max-steps 900 --step-token-budget 1 \
    --vllm-gpu-memory-utilization 0.30 \
    --gsm-split-file "$GSPLIT" --gsm-split-name eval \
    --output-json outputs/controlled_comparison/gsm_1p5B/metadecode.json
  echo "EXIT_REEVAL_GSM1P5B=$?"
else
  echo "NO_ACCEPTED_CSD_GSM1P5B"
fi
echo DONE_GSM1P5B_CHAIN
