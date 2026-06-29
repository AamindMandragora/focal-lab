#!/bin/bash
# Targeted experiment: re-run att14's EXACT compiled strategy against the FIXED
# CloseConstrainedSpan library, on the same 50 GSM examples / same eval config.
#
# Inputs:  att14 build dir (deterministically = ..._201725_36664d, attempt_ts+6s)
#          + fixed VerifiedDecoderAgent.py (rendered-text-suffix dedup).
# Output:  accuracy + syntax_rate over 50 examples -> /tmp/att14_fixed_result.json
# Expect:  syntax 0.76 -> ~0.92 (the 8 doubled-'>>' cases now close with one '>>'),
#          accuracy held ~0.56 (inner math unchanged).
set -u
cd /home/aadivyar/csd-generation

ATT14_BUILD=outputs/generated/ralph_1p5B_gsm_relaunch_20260529_detector/ralph_1p5B_gsm_relaunch_20260529_detector_20260529_182754_657382/python/ralph_1p5B_gsm_relaunch_20260529_detector_20260529_201725_36664d
FIXED_LIB=/tmp/fixed_build/crane-py/VerifiedDecoderAgent.py
WORK=/tmp/att14_fixed

rm -rf "$WORK"
cp -r "$ATT14_BUILD" "$WORK"
cp "$FIXED_LIB" "$WORK/VerifiedDecoderAgent.py"

echo "=== sanity: fixed lib in place? ==="
echo "RenderedEndsWith count in WORK lib: $(grep -c RenderedEndsWith "$WORK/VerifiedDecoderAgent.py")"
echo "att14 GeneratedCSD.py present: $(test -f "$WORK/GeneratedCSD.py" && echo yes || echo NO)"

export PYTHONPATH=/home/aadivyar/csd-generation
export CUDA_VISIBLE_DEVICES=2
export LD_LIBRARY_PATH=/apps/conda/advayth2/envs/advayth2/lib:${LD_LIBRARY_PATH:-}
export VLLM_WORKER_MULTIPROC_METHOD=spawn

/apps/conda/advayth2/envs/advayth2/bin/python synthesis/scripts/reevaluate_compiled_csd.py \
  "$WORK/GeneratedCSD.py" \
  --dataset gsm_symbolic \
  --eval-model Qwen/Qwen2.5-1.5B-Instruct --eval-backend vllm \
  --device cuda --sample-size 50 --max-steps 900 --step-token-budget 1 \
  --vllm-gpu-memory-utilization 0.40 --vllm-tensor-parallel-size 1 \
  --gsm-split-file /home/aadivyar/csd-generation/environment/benchmark_splits/gsm_symbolic_crane_proportional.json \
  --gsm-split-name eval \
  --max-seconds-per-example 90 \
  --output-json /tmp/att14_fixed_result.json
echo "REEVAL_EXIT=$?"
