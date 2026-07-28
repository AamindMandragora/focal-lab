#!/usr/bin/env bash
# Rerun SMILES CARS baselines and guard metaDecode against bare-output regressions.
#
# This script does not overwrite old results. New outputs go under:
#   outputs/controlled_comparison_bare_smiles/
#
# If a metaDecode re-eval under the bare SMILES contract scores lower than the
# prior held-out JSON, the script appends a COLD synthesis command to:
#   .context/smiles_bare_prompt_paid_synthesis_todo.sh
#
# Do not run the paid todo file without fresh billing confirmation.
set -uo pipefail

REPO="${REPO:-/home/aadivyar/csd-generation}"
PY="${PY:-/apps/conda/aadivyar/envs/csd/bin/python}"
GPU="${GPU:-2}"
UTIL="${UTIL:-0.45}"
export CUDA_VISIBLE_DEVICES="$GPU"
export LD_LIBRARY_PATH=/apps/conda/aadivyar/envs/csd/lib:${LD_LIBRARY_PATH:-}
export HF_HOME=/home/aadivyar/.cache/huggingface
export TRANSFORMERS_CACHE=/home/aadivyar/.cache/huggingface

cd "$REPO"
mkdir -p logs .context outputs/controlled_comparison_bare_smiles

STATUS="logs/smiles_bare_prompt_queue_status.tsv"
DRIVER_LOG="logs/smiles_bare_prompt_queue_driver.log"
PAID_TODO=".context/smiles_bare_prompt_paid_synthesis_todo.sh"

if [ ! -s "$STATUS" ]; then
  printf "kind\tlabel\tstatus\told_json\tnew_json\told_acc\tnew_acc\texit_code\tfinished_at\n" > "$STATUS"
fi

if [ ! -s "$PAID_TODO" ]; then
  cat > "$PAID_TODO" <<'TODO'
#!/usr/bin/env bash
# Auto-filled by run_smiles_bare_prompt_queue.sh.
# DO NOT RUN without fresh billing confirmation.
set -uo pipefail
cd /home/aadivyar/csd-generation
TODO
fi

log() {
  echo "[smiles-bare-queue] $* $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$DRIVER_LOG"
}

new_path_for() {
  local old="$1"
  echo "outputs/controlled_comparison_bare_smiles/${old#outputs/controlled_comparison/}"
}

run_cars() {
  local label="$1" model="$2" class_name="$3" sample_size="$4" old_json="$5"
  local out_json
  out_json="$(new_path_for "$old_json")"
  mkdir -p "$(dirname "$out_json")"
  if [ -s "$out_json" ]; then
    log "skip existing CARS $label -> $out_json"
    printf "cars\t%s\tskipped-existing\t%s\t%s\t\t\t0\t%s\n" \
      "$label" "$old_json" "$out_json" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS"
    return 0
  fi

  log "start CARS $label -> $out_json"
  "$PY" -m synthesis.evaluate.run_legacy_fixed_strategy \
    --strategy cars \
    --dataset smiles \
    --eval-model "$model" \
    --eval-backend vllm \
    --device auto \
    --smiles-classes "$class_name" \
    --eval-sample-size "$sample_size" \
    --eval-max-steps 400 \
    --eval-step-token-budget 1 \
    --cars-search-steps 200 \
    --vllm-gpu-memory-utilization "$UTIL" \
    --output-json "$out_json"
  local ec=$?
  printf "cars\t%s\tfinished\t%s\t%s\t\t\t%s\t%s\n" \
    "$label" "$old_json" "$out_json" "$ec" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS"
  return "$ec"
}

json_number() {
  "$PY" - "$1" "$2" <<'PY'
import json, sys
path, key = sys.argv[1], sys.argv[2]
try:
    value = json.load(open(path)).get(key)
except Exception:
    value = None
print("" if value is None else value)
PY
}

compiled_csd_for() {
  local output_name="$1"
  local latest="outputs/generated/$output_name/latest_run.txt"
  [ -s "$latest" ] || return 1
  local run_dir
  run_dir="$(cat "$latest")"
  [ -n "$run_dir" ] || return 1
  local report="$run_dir/results/success_report.json"
  [ -s "$report" ] || return 1
  local compiled_dir
  compiled_dir="$("$PY" - "$report" <<'PY'
import json, sys
print(json.load(open(sys.argv[1])).get("compiled_dir", ""))
PY
)"
  [ -n "$compiled_dir" ] || return 1
  [ -s "$compiled_dir/GeneratedCSD.py" ] || return 1
  echo "$compiled_dir/GeneratedCSD.py"
}

append_paid_synthesis_if_lower() {
  local label="$1" tag="$2" model="$3" class_name="$4" old_json="$5" new_json="$6" output_name="$7"
  local old_acc new_acc old_syn
  old_acc="$(json_number "$old_json" accuracy)"
  new_acc="$(json_number "$new_json" accuracy)"
  old_syn="$(json_number "$old_json" syntax_rate)"
  "$PY" - "$old_acc" "$new_acc" <<'PY'
import sys
old = float(sys.argv[1])
new = float(sys.argv[2])
raise SystemExit(0 if new + 1e-12 < old else 1)
PY
  local lower=$?
  if [ "$lower" -ne 0 ]; then
    log "metaDecode unchanged-or-better $label old=$old_acc new=$new_acc"
    return 0
  fi

  local synth_name="smiles_bare_${tag}_${class_name}_recover_$(date -u +%Y%m%d)"
  log "queue paid COLD synthesis $label old=$old_acc new=$new_acc -> $synth_name"
  cat >> "$PAID_TODO" <<TODO

# Queued because bare-output held-out re-eval dropped: $label old=$old_acc new=$new_acc.
# Requires fresh billing confirmation before launch.
SMILES_TASK='Generate one new, valid, non-exemplar SMILES molecule for the ${class_name} class. The answer contract is a single SMILES string and nothing else. Use the hidden parser-guided constrained chunk for that SMILES token sequence and avoid copying prompt exemplars.' \\
DATASET=smiles SMILES_CLASS=${class_name} EVAL_MODEL='${model}' GPU="\${GPU:-3}" GPU_MEM_UTIL=0.45 \\
MIN_ACC=${old_acc} MIN_SYN=${old_syn:-0.0} MAX_ITERS=40 SAMPLE_SIZE=50 EVAL_MAX_STEPS=400 \\
OUTPUT_NAME='${synth_name}' bash run_synth_cell.sh
TODO
}

run_metadecode_guard() {
  local label="$1" tag="$2" model="$3" class_name="$4" old_json="$5" output_name="$6"
  local out_json
  out_json="$(new_path_for "$old_json")"
  mkdir -p "$(dirname "$out_json")"
  if [ ! -s "$old_json" ]; then
    log "skip missing old metaDecode JSON $label -> $old_json"
    return 0
  fi
  if [ -s "$out_json" ]; then
    log "skip existing metaDecode re-eval $label -> $out_json"
  else
    local csd
    if ! csd="$(compiled_csd_for "$output_name")"; then
      log "skip no accepted CSD for $label output_name=$output_name"
      printf "metadecode\t%s\tmissing-csd\t%s\t%s\t\t\t1\t%s\n" \
        "$label" "$old_json" "$out_json" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS"
      return 0
    fi
    log "start metaDecode bare re-eval $label -> $out_json"
    "$PY" -m synthesis.scripts.reevaluate_compiled_csd "$csd" \
      --dataset smiles \
      --smiles-classes "$class_name" \
      --eval-model "$model" \
      --eval-backend vllm \
      --sample-size 100 \
      --max-steps 400 \
      --step-token-budget 1 \
      --vllm-gpu-memory-utilization "$UTIL" \
      --output-json "$out_json"
    local ec=$?
    local old_acc new_acc
    old_acc="$(json_number "$old_json" accuracy)"
    new_acc="$(json_number "$out_json" accuracy)"
    printf "metadecode\t%s\tfinished\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$label" "$old_json" "$out_json" "$old_acc" "$new_acc" "$ec" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS"
    [ "$ec" -eq 0 ] || return "$ec"
  fi
  append_paid_synthesis_if_lower "$label" "$tag" "$model" "$class_name" "$old_json" "$out_json" "$output_name"
}

log "queue start GPU=$GPU util=$UTIL"

# Recorded SMILES CARS controlled-comparison outputs.
run_cars "qwen25-1p5b-acrylates-n100" "Qwen/Qwen2.5-1.5B-Instruct" "acrylates" 100 "outputs/controlled_comparison/smiles_1p5B/acrylates/cars.json"
run_cars "qwen25-1p5b-chain-extenders-n100" "Qwen/Qwen2.5-1.5B-Instruct" "chain_extenders" 100 "outputs/controlled_comparison/smiles_1p5B/chain_extenders/cars.json"
run_cars "qwen25-1p5b-isocyanates-n100" "Qwen/Qwen2.5-1.5B-Instruct" "isocyanates" 100 "outputs/controlled_comparison/smiles_1p5B/isocyanates/cars.json"
run_cars "qwen25-7b-acrylates-n100" "Qwen/Qwen2.5-7B-Instruct" "acrylates" 100 "outputs/controlled_comparison/smiles_7B/acrylates/cars.json"
run_cars "qwen25-7b-chain-extenders-n100" "Qwen/Qwen2.5-7B-Instruct" "chain_extenders" 100 "outputs/controlled_comparison/smiles_7B/chain_extenders/cars.json"
run_cars "qwen25-7b-isocyanates-n100" "Qwen/Qwen2.5-7B-Instruct" "isocyanates" 100 "outputs/controlled_comparison/smiles_7B/isocyanates/cars.json"

run_cars "qwen35-2b-acrylates-n50" "Qwen/Qwen3.5-2B" "acrylates" 50 "outputs/controlled_comparison/smiles_qwen35/2B/acrylates/cars.json"
run_cars "qwen35-2b-chain-extenders-n50" "Qwen/Qwen3.5-2B" "chain_extenders" 50 "outputs/controlled_comparison/smiles_qwen35/2B/chain_extenders/cars.json"
run_cars "qwen35-2b-isocyanates-n50" "Qwen/Qwen3.5-2B" "isocyanates" 50 "outputs/controlled_comparison/smiles_qwen35/2B/isocyanates/cars.json"
run_cars "qwen35-4b-acrylates-n50" "Qwen/Qwen3.5-4B" "acrylates" 50 "outputs/controlled_comparison/smiles_qwen35/4B/acrylates/cars.json"
run_cars "qwen35-4b-chain-extenders-n50" "Qwen/Qwen3.5-4B" "chain_extenders" 50 "outputs/controlled_comparison/smiles_qwen35/4B/chain_extenders/cars.json"
run_cars "qwen35-4b-isocyanates-n50" "Qwen/Qwen3.5-4B" "isocyanates" 50 "outputs/controlled_comparison/smiles_qwen35/4B/isocyanates/cars.json"
run_cars "qwen35-9b-acrylates-n50" "Qwen/Qwen3.5-9B" "acrylates" 50 "outputs/controlled_comparison/smiles_qwen35/9B/acrylates/cars.json"
run_cars "qwen35-9b-chain-extenders-n50" "Qwen/Qwen3.5-9B" "chain_extenders" 50 "outputs/controlled_comparison/smiles_qwen35/9B/chain_extenders/cars.json"
run_cars "qwen35-9b-isocyanates-n50" "Qwen/Qwen3.5-9B" "isocyanates" 50 "outputs/controlled_comparison/smiles_qwen35/9B/isocyanates/cars.json"

run_cars "qwen35-2b-acrylates-n100" "Qwen/Qwen3.5-2B" "acrylates" 100 "outputs/controlled_comparison/smiles_qwen35_2b/acrylates/cars.json"
run_cars "qwen35-2b-chain-extenders-n100" "Qwen/Qwen3.5-2B" "chain_extenders" 100 "outputs/controlled_comparison/smiles_qwen35_2b/chain_extenders/cars.json"
run_cars "qwen35-2b-isocyanates-n100" "Qwen/Qwen3.5-2B" "isocyanates" 100 "outputs/controlled_comparison/smiles_qwen35_2b/isocyanates/cars.json"
run_cars "qwen35-4b-acrylates-n100" "Qwen/Qwen3.5-4B" "acrylates" 100 "outputs/controlled_comparison/smiles_qwen35_4b/acrylates/cars.json"
run_cars "qwen35-4b-chain-extenders-n100" "Qwen/Qwen3.5-4B" "chain_extenders" 100 "outputs/controlled_comparison/smiles_qwen35_4b/chain_extenders/cars.json"
run_cars "qwen35-4b-isocyanates-n100" "Qwen/Qwen3.5-4B" "isocyanates" 100 "outputs/controlled_comparison/smiles_qwen35_4b/isocyanates/cars.json"
run_cars "qwen35-9b-acrylates-n100" "Qwen/Qwen3.5-9B" "acrylates" 100 "outputs/controlled_comparison/smiles_qwen35_9b/acrylates/cars.json"
run_cars "qwen35-9b-chain-extenders-n100" "Qwen/Qwen3.5-9B" "chain_extenders" 100 "outputs/controlled_comparison/smiles_qwen35_9b/chain_extenders/cars.json"
run_cars "qwen35-9b-isocyanates-n100" "Qwen/Qwen3.5-9B" "isocyanates" 100 "outputs/controlled_comparison/smiles_qwen35_9b/isocyanates/cars.json"
run_cars "qwen35-4b-acrylates-sanity-n3" "Qwen/Qwen3.5-4B" "acrylates" 3 "outputs/controlled_comparison/smiles_qwen35_SANITY4B/4B/acrylates/cars.json"

# Existing Qwen3.5 metaDecode held-out cells with accepted CSDs on disk.
run_metadecode_guard "qwen35-2b-acrylates" "qwen35_2b" "Qwen/Qwen3.5-2B" "acrylates" "outputs/controlled_comparison/smiles_qwen35_2b/acrylates/metadecode_uv.json" "smiles_qwen35_2b_acrylates_uv_qwen35_0627"
run_metadecode_guard "qwen35-2b-chain-extenders" "qwen35_2b" "Qwen/Qwen3.5-2B" "chain_extenders" "outputs/controlled_comparison/smiles_qwen35_2b/chain_extenders/metadecode_uv.json" "smiles_qwen35_2b_chain_extenders_uv_qwen35_0627"
run_metadecode_guard "qwen35-2b-isocyanates" "qwen35_2b" "Qwen/Qwen3.5-2B" "isocyanates" "outputs/controlled_comparison/smiles_qwen35_2b/isocyanates/metadecode_uv.json" "smiles_qwen35_2b_isocyanates_uv_qwen35_0627"
run_metadecode_guard "qwen35-4b-isocyanates" "qwen35_4b" "Qwen/Qwen3.5-4B" "isocyanates" "outputs/controlled_comparison/smiles_qwen35_4b/isocyanates/metadecode_uv.json" "smiles_qwen35_4b_isocyanates_uv_qwen35_0627"
run_metadecode_guard "qwen35-9b-acrylates" "qwen35_9b" "Qwen/Qwen3.5-9B" "acrylates" "outputs/controlled_comparison/smiles_qwen35_9b/acrylates/metadecode_uv.json" "smiles_qwen35_9b_acrylates_uv_qwen35_0627"
run_metadecode_guard "qwen35-9b-isocyanates" "qwen35_9b" "Qwen/Qwen3.5-9B" "isocyanates" "outputs/controlled_comparison/smiles_qwen35_9b/isocyanates/metadecode_uv.json" "smiles_qwen35_9b_isocyanates_uv_qwen35_0627"

log "queue done. Paid todo, if any: $PAID_TODO"
