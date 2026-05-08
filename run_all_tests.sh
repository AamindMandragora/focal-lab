#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

DEFAULT_MODELS="Qwen/Qwen2.5-Coder-1.5B-Instruct,Qwen/Qwen2.5-Coder-7B-Instruct,Qwen/Qwen2.5-Coder-14B-Instruct"
DEFAULT_BENCHMARKS="smiles,gsm,spider"
DEFAULT_STRATEGIES="unconstrained,crane,itergen,cars,metadecode"
DEFAULT_TOKEN_BUDGETS="1,2,4"
DEFAULT_SYNTH_ITERS="3,5,10"
DEFAULT_GEN_MODELS="gpt5.4,opus4.7,gemini-pro"

MODELS="$DEFAULT_MODELS"
BENCHMARKS="$DEFAULT_BENCHMARKS"
STRATEGIES="$DEFAULT_STRATEGIES"
TOKEN_BUDGETS="$DEFAULT_TOKEN_BUDGETS"
SYNTH_ITERS="$DEFAULT_SYNTH_ITERS"
GEN_MODELS="$DEFAULT_GEN_MODELS"

EVAL_BACKEND="vllm"
DEVICE="auto"
EVAL_SAMPLE_SIZE="10"
EVAL_MAX_STEPS="150"
VLLM_GPU_MEM_UTIL="0.8"
DAFNY_PATH="$ROOT_DIR/dafny/dafny"
GENERATED_OUTPUT_DIR="${CSD_OUTPUT_DIR:-outputs/generated}"
BASELINE_OUTPUT_DIR="${CSD_BASELINE_OUTPUT_DIR:-outputs/baselines}"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: ./run_all_tests.sh [options]

Runs strategy x model x benchmark matrix, plus ablations.
Without arguments, defaults to:
- eval models: Qwen2.5-Coder {1.5B, 7B, 14B}
- benchmarks: smiles, gsm, spider
- strategies: unconstrained, crane, itergen, cars, metadecode
- eval token-budget ablation: 1,2,4
- metadecode synthesis-iteration ablation: 3,5,10
- metadecode generation-model ablation: gpt5.4, opus4.7, gemini-pro

Options:
  --models CSV                  Eval models list
  --benchmarks CSV              Benchmarks list (gsm,gsm_symbolic,spider,smiles)
  --strategies CSV              Strategies list (unconstrained,crane,itergen,cars,metadecode)
  --token-budgets CSV           Eval step token budgets
  --synthesis-iterations CSV    Metadecode synthesis-iteration ablation values
  --generation-models CSV       Metadecode synthesis generation model profiles
  --eval-backend NAME           huggingface|vllm (default: vllm)
  --device NAME                 auto|cuda|cpu|mps (default: auto)
  --eval-sample-size N          Evaluation sample size (default: 10)
  --eval-max-steps N            Eval max steps (default: 150)
  --vllm-gpu-memory-utilization FLOAT
  --dafny-path PATH
  --generated-output-dir PATH    Synthesis output directory (default: outputs/generated/ or CSD_OUTPUT_DIR)
  --baseline-output-dir PATH     Baseline JSON directory (default: outputs/baselines/ or CSD_BASELINE_OUTPUT_DIR)
  --dry-run                     Print commands only
  -h, --help                    Show help

Outputs:
- Synthesis runs: \$GENERATED_OUTPUT_DIR
- Baseline JSONs: \$BASELINE_OUTPUT_DIR
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models) MODELS="$2"; shift 2 ;;
    --benchmarks) BENCHMARKS="$2"; shift 2 ;;
    --strategies) STRATEGIES="$2"; shift 2 ;;
    --token-budgets) TOKEN_BUDGETS="$2"; shift 2 ;;
    --synthesis-iterations) SYNTH_ITERS="$2"; shift 2 ;;
    --generation-models) GEN_MODELS="$2"; shift 2 ;;
    --eval-backend) EVAL_BACKEND="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --eval-sample-size) EVAL_SAMPLE_SIZE="$2"; shift 2 ;;
    --eval-max-steps) EVAL_MAX_STEPS="$2"; shift 2 ;;
    --vllm-gpu-memory-utilization) VLLM_GPU_MEM_UTIL="$2"; shift 2 ;;
    --dafny-path) DAFNY_PATH="$2"; shift 2 ;;
    --generated-output-dir) GENERATED_OUTPUT_DIR="$2"; shift 2 ;;
    --baseline-output-dir) BASELINE_OUTPUT_DIR="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

mkdir -p "$GENERATED_OUTPUT_DIR" "$BASELINE_OUTPUT_DIR"

IFS=',' read -r -a MODELS_ARR <<< "$MODELS"
IFS=',' read -r -a BENCHMARKS_ARR <<< "$BENCHMARKS"
IFS=',' read -r -a STRATEGIES_ARR <<< "$STRATEGIES"
IFS=',' read -r -a TOKEN_BUDGETS_ARR <<< "$TOKEN_BUDGETS"
IFS=',' read -r -a SYNTH_ITERS_ARR <<< "$SYNTH_ITERS"
IFS=',' read -r -a GEN_MODELS_ARR <<< "$GEN_MODELS"

normalize_benchmark() {
  local b="$1"
  case "$b" in
    gsm) echo "gsm_symbolic" ;;
    gsm_symbolic|spider|smiles) echo "$b" ;;
    *) echo "$b" ;;
  esac
}

slugify() {
  local s="$1"
  s="${s//\//_}"
  s="${s//:/_}"
  s="${s// /_}"
  s="${s//-/_}"
  echo "$s"
}

metadecode_task() {
  local benchmark="$1"
  case "$benchmark" in
    gsm_symbolic)
      echo "Solve math word problems step by step, writing each arithmetic computation inside << >> delimiters."
      ;;
    spider)
      echo "Generate a single valid SQL query that answers each question using the provided schema context."
      ;;
    smiles)
      echo "Generate valid SMILES strings that match the requested molecular class while maintaining parser-valid output."
      ;;
    *)
      echo "Generate parser-valid benchmark answers."
      ;;
  esac
}

resolve_gen_profile() {
  local profile="$1"
  case "$profile" in
    gpt5.4)
      echo "openai|gpt-5.4"
      ;;
    opus4.7)
      echo "anthropic|claude-opus-4-7"
      ;;
    gemini-pro)
      echo "gemini|gemini-2.5-pro"
      ;;
    *)
      echo "openai|$profile"
      ;;
  esac
}

run_cmd() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] $*"
    return 0
  fi
  echo "[run] $*"
  "$@"
}

run_fixed_strategy_case() {
  local strategy="$1"
  local benchmark="$2"
  local eval_model="$3"
  local token_budget="$4"

  local model_slug
  model_slug="$(slugify "$eval_model")"

  local out_json="${BASELINE_OUTPUT_DIR}/${strategy}/${model_slug}/${benchmark}__tb${token_budget}.json"
  mkdir -p "$(dirname "$out_json")"

  local cmd=(
    python -m synthesis.evaluate.run_legacy_fixed_strategy
    --strategy "$strategy"
    --dataset "$benchmark"
    --eval-model "$eval_model"
    --eval-backend "$EVAL_BACKEND"
    --device "$DEVICE"
    --eval-sample-size "$EVAL_SAMPLE_SIZE"
    --eval-max-steps "$EVAL_MAX_STEPS"
    --eval-step-token-budget "$token_budget"
    --vllm-gpu-memory-utilization "$VLLM_GPU_MEM_UTIL"
    --output-json "$out_json"
  )

  if [[ -n "$DAFNY_PATH" ]]; then
    cmd+=(--dafny-path "$DAFNY_PATH")
  fi

  run_cmd "${cmd[@]}"
}

run_metadecode_case() {
  local benchmark="$1"
  local eval_model="$2"
  local token_budget="$3"
  local synth_iter="$4"
  local gen_profile="$5"

  local resolved backend generation_model
  resolved="$(resolve_gen_profile "$gen_profile")"
  backend="${resolved%%|*}"
  generation_model="${resolved##*|}"

  local model_slug gen_slug run_name task
  model_slug="$(slugify "$eval_model")"
  gen_slug="$(slugify "$gen_profile")"
  run_name="metadecode_${benchmark}_${model_slug}_${gen_slug}_iter${synth_iter}_tb${token_budget}"
  task="$(metadecode_task "$benchmark")"

  local synth_cmd=(
    python -m synthesis.run_synthesis
    --task "$task"
    --dataset "$benchmark"
    --generation-model "$generation_model"
    --generation-backend "$backend"
    --eval-model "$eval_model"
    --eval-backend "$EVAL_BACKEND"
    --max-iterations "$synth_iter"
    --output-name "$run_name"
    --min-accuracy "0.0"
    --min-syntax-rate "0.0"
    --eval-sample-size "$EVAL_SAMPLE_SIZE"
    --eval-max-steps "$EVAL_MAX_STEPS"
    --eval-step-token-budget "$token_budget"
    --vllm-gpu-memory-utilization "$VLLM_GPU_MEM_UTIL"
    --device "$DEVICE"
    --output-dir "$GENERATED_OUTPUT_DIR"
  )

  if [[ -n "$DAFNY_PATH" ]]; then
    synth_cmd+=(--dafny-path "$DAFNY_PATH")
  fi

  if ! run_cmd "${synth_cmd[@]}"; then
    echo "[warn] Metadecode synthesis failed for benchmark=$benchmark eval_model=$eval_model token_budget=$token_budget iter=$synth_iter gen=$gen_profile" >&2
    return 0
  fi

  local out_json
  out_json="${BASELINE_OUTPUT_DIR}/metadecode/${model_slug}/${benchmark}__tb${token_budget}__gen${gen_slug}__iter${synth_iter}.json"
  mkdir -p "$(dirname "$out_json")"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] python -m synthesis.evaluate.export_baseline_json --success-report <${GENERATED_OUTPUT_DIR}/.../results/success_report.json> --output $out_json"
    return 0
  fi

  local latest_file run_dir success_report
  latest_file="${GENERATED_OUTPUT_DIR}/latest_run.txt"
  if [[ ! -f "$latest_file" ]]; then
    echo "[warn] No latest run file found after synthesis: $latest_file" >&2
    return 0
  fi

  run_dir="$(tr -d '\n' < "$latest_file")"
  success_report="$run_dir/results/success_report.json"
  if [[ ! -f "$success_report" ]]; then
    echo "[warn] No success report found for run: $run_dir" >&2
    return 0
  fi

  run_cmd python -m synthesis.evaluate.export_baseline_json \
    --success-report "$success_report" \
    --output "$out_json"
}

echo "=== run_all_tests matrix ==="
echo "models: ${MODELS_ARR[*]}"
echo "benchmarks: ${BENCHMARKS_ARR[*]}"
echo "strategies: ${STRATEGIES_ARR[*]}"
echo "token budgets: ${TOKEN_BUDGETS_ARR[*]}"
echo "synthesis iters (metadecode): ${SYNTH_ITERS_ARR[*]}"
echo "generation models (metadecode): ${GEN_MODELS_ARR[*]}"

for raw_benchmark in "${BENCHMARKS_ARR[@]}"; do
  benchmark="$(normalize_benchmark "$raw_benchmark")"
  for eval_model in "${MODELS_ARR[@]}"; do
    for strategy in "${STRATEGIES_ARR[@]}"; do
      if [[ "$strategy" == "metadecode" ]]; then
        for token_budget in "${TOKEN_BUDGETS_ARR[@]}"; do
          for synth_iter in "${SYNTH_ITERS_ARR[@]}"; do
            for gen_profile in "${GEN_MODELS_ARR[@]}"; do
              run_metadecode_case "$benchmark" "$eval_model" "$token_budget" "$synth_iter" "$gen_profile"
            done
          done
        done
      else
        for token_budget in "${TOKEN_BUDGETS_ARR[@]}"; do
          run_fixed_strategy_case "$strategy" "$benchmark" "$eval_model" "$token_budget"
        done
      fi
    done
  done
done

echo "All requested matrix jobs completed."
