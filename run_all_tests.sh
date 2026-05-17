#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ -f "$ROOT_DIR/synthesis/.env" ]]; then
  set -a
  source "$ROOT_DIR/synthesis/.env"
  set +a
fi

# Activate the conda environment that has RDKit (needed for SMILES evaluation).
# Default: shared lab prefix. Override for another compatible env:
#   export VAS_CONDA_ENV="/path/to/your/env"    # preferred name
#   export VAS_RDKIT_CONDA_ENV="$VAS_CONDA_ENV" # legacy alias (still honored)
_DEFAULT_VAS_CONDA_ENV="/apps/conda/advayth2/envs/advayth2"
CONDA_ENV_PATH="${VAS_CONDA_ENV:-${VAS_RDKIT_CONDA_ENV:-${_DEFAULT_VAS_CONDA_ENV}}}"
if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required to run the matrix in $CONDA_ENV_PATH" >&2
  exit 1
fi
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_PATH"
_expected="$(cd "$CONDA_ENV_PATH" && pwd -P)"
_actual="$(cd "${CONDA_PREFIX:-}" && pwd -P 2>/dev/null || echo "")"
if [[ -z "$_actual" || "$_actual" != "$_expected" ]]; then
  echo "failed to activate required conda environment: $CONDA_ENV_PATH" >&2
  echo "current CONDA_PREFIX (canonical): ${_actual:-<unset>}" >&2
  echo "expected canonical path: $_expected" >&2
  exit 1
fi
python - <<'PY'
import rdkit
PY
echo "[env] using conda environment: $CONDA_PREFIX"

# SciPy / PyTorch wheels link against a newer libstdc++ than some distro defaults expose.
# Prepend conda's runtime libs so imports resolve CXXABI (e.g. CXXABI_1.3.15) correctly.
if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/lib" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

# Help redirected logs show tqdm/Python subprocess lines promptly during matrix runs.
export PYTHONUNBUFFERED=1

# Pin visible GPUs after sourcing synthesis/.env (it may set busy cards 0/1).
# Default 3 dedicates physical GPU 3 to this matrix (process sees it as cuda:0).
# Shared-node preset: RUN_ALL_TESTS_CUDA_DEVICES=2,3 ./run_all_tests.sh
RUN_ALL_TESTS_CUDA_DEVICES="${RUN_ALL_TESTS_CUDA_DEVICES:-3}"
export CUDA_VISIBLE_DEVICES="$RUN_ALL_TESTS_CUDA_DEVICES"
# On CUDA OOM, matrix subprocesses retry once with this visibility (unset defaults to 2; empty string disables).
RUN_ALL_TESTS_CUDA_OOM_FALLBACK="${RUN_ALL_TESTS_CUDA_OOM_FALLBACK-2}"
echo "[env] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
if [[ -n "${RUN_ALL_TESTS_CUDA_OOM_FALLBACK:-}" ]]; then
  echo "[env] RUN_ALL_TESTS_CUDA_OOM_FALLBACK=$RUN_ALL_TESTS_CUDA_OOM_FALLBACK (OOM retry; set empty to disable)"
else
  echo "[env] RUN_ALL_TESTS_CUDA_OOM_FALLBACK unset/disabled"
fi

DEFAULT_MODELS="Qwen/Qwen2.5-Coder-1.5B-Instruct,Qwen/Qwen2.5-Coder-7B-Instruct,Qwen/Qwen2.5-Coder-14B-Instruct,meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_BENCHMARKS="gsm,spider,smiles"
DEFAULT_STRATEGIES="unconstrained,gcd,crane,itergen,cars,metadecode"
DEFAULT_TOKEN_BUDGETS="1,2,4"
DEFAULT_SYNTH_ITERS="3,5,10"
DEFAULT_GEN_MODELS="gpt5.4,opus4.7"
DEFAULT_STEP_BUDGETS="256,512,1024"
DEFAULT_SMILES_CLASSES="acrylates,chain_extenders,isocyanates"

MODELS="$DEFAULT_MODELS"
BENCHMARKS="$DEFAULT_BENCHMARKS"
STRATEGIES="$DEFAULT_STRATEGIES"
TOKEN_BUDGETS="$DEFAULT_TOKEN_BUDGETS"
SYNTH_ITERS="$DEFAULT_SYNTH_ITERS"
GEN_MODELS="$DEFAULT_GEN_MODELS"
STEP_BUDGETS="$DEFAULT_STEP_BUDGETS"
SMILES_CLASSES="$DEFAULT_SMILES_CLASSES"

EVAL_BACKEND="vllm"
DEVICE="auto"
EVAL_SAMPLE_SIZE="100"
EVAL_MAX_STEPS="900"
# Higher default when pinning a single card so vLLM reserves most VRAM on that GPU.
VLLM_GPU_MEM_UTIL="0.95"
DAFNY_PATH="${DAFNY_PATH:-}"
if [[ -z "$DAFNY_PATH" && -x "$ROOT_DIR/dafny/dafny" ]]; then
  DAFNY_PATH="$ROOT_DIR/dafny/dafny"
fi
GENERATED_OUTPUT_DIR="${CSD_OUTPUT_DIR:-outputs/generated}"
BASELINE_OUTPUT_DIR="${CSD_BASELINE_OUTPUT_DIR:-outputs/baselines}"
ABLATION_OUTPUT_DIR="${CSD_ABLATION_OUTPUT_DIR:-outputs/ablations}"
BASELINE_CACHE_MODE="${CSD_BASELINE_CACHE_MODE:-reuse}"
GSM_SPLIT_FILE=""
SPIDER_SPLIT_FILE=""
DRY_RUN=0
SKIP_MAIN=0
SKIP_ABLATIONS=0

usage() {
  cat <<'EOF'
Usage: ./run_all_tests.sh [options]

Runs strategy x model x benchmark matrix, plus ablations.
Without arguments, defaults to:
- eval models: Qwen2.5-Coder {1.5B, 7B, 14B}, Llama-3.1-8B-Instruct
- benchmarks: gsm, spider, smiles
- strategies: unconstrained, gcd, crane, itergen, cars, metadecode
- eval token-budget ablation: 1,2,4
- metadecode synthesis-iteration ablation: 3,5,10
- metadecode generation-model ablation: gpt5.4, opus4.7 (gemini-pro omitted until wired)
- step-budget ablation: 256,512,1024

All strategies are evaluated on all benchmarks via legacy external
codebases (crane, itergen, cars, gcd/syncode) or the synthesis
pipeline (metadecode). Unconstrained uses the legacy adapter.

Environment:
  Default conda prefix /apps/conda/advayth2/envs/advayth2 (must include RDKit).
  Override with VAS_CONDA_ENV or legacy VAS_RDKIT_CONDA_ENV (see environment/README.md).
  Prepends CONDA_PREFIX/lib to LD_LIBRARY_PATH for SciPy/transformers wheels vs system libstdc++.
  Sets CUDA_VISIBLE_DEVICES (default 3) after loading synthesis/.env; override with RUN_ALL_TESTS_CUDA_DEVICES (e.g. 2,3).
  Optional RUN_ALL_TESTS_CUDA_OOM_FALLBACK (default 2): each GPU job tries the primary visibility first; if logs indicate CUDA OOM, retries once with the fallback visibility. Set RUN_ALL_TESTS_CUDA_OOM_FALLBACK= to disable.
  Metadecode: profile gpt5.4 uses OpenAI (--generation-backend openai, OPENAI_API_KEY); opus4.7 uses Bedrock (AWS_BEARER_TOKEN_BEDROCK, BEDROCK_OPUS_MODEL). Profile gemini-pro is disabled in the default GEN_MODELS list; set GEMINI_BEDROCK_MODEL if you pass gemini-pro explicitly.

Options:
  --models CSV                  Eval models list
  --benchmarks CSV              Benchmarks list (gsm,gsm_symbolic,spider,smiles)
  --strategies CSV              Strategies list (unconstrained,gcd,crane,itergen,cars,metadecode)
  --token-budgets CSV           Eval step token budgets (per-step)
  --step-budgets CSV            Max-steps budget ablation values (total generation budget)
  --synthesis-iterations CSV    Metadecode synthesis-iteration ablation values
  --generation-models CSV       Metadecode synthesis generation model profiles
  --smiles-classes CSV          SMILES classes to evaluate; each run gets one class-specific grammar
                                (default: acrylates,chain_extenders,isocyanates)
  --eval-backend NAME           huggingface|vllm (default: vllm)
  --device NAME                 auto|cuda|cpu|mps (default: auto)
  --eval-sample-size N          Evaluation sample size for baselines and metadecode (default: 100)
  --eval-max-steps N            Eval max steps for main matrix (default: 900)
  --gsm-split-file PATH         Train/eval split manifest for GSM (disjoint splits)
  --spider-split-file PATH      Train/eval split manifest for Spider (disjoint splits)
  --vllm-gpu-memory-utilization FLOAT
  --dafny-path PATH
  --generated-output-dir PATH    Synthesis output directory (default: outputs/generated/ or CSD_OUTPUT_DIR)
  --baseline-output-dir PATH     Baseline JSON directory (default: outputs/baselines/ or CSD_BASELINE_OUTPUT_DIR)
  --ablation-output-dir PATH     Ablation JSON directory (default: outputs/ablations/)
  --recompute-baselines          Re-run fixed-strategy baselines instead of reusing cached JSONs
  --reuse-baselines              Reuse complete cached baseline JSONs (default)
  --skip-main                   Skip main matrix, run ablations only
  --skip-ablations              Skip ablations, run main matrix only
  --dry-run                     Print commands only
  -h, --help                    Show help

Outputs:
- Synthesis runs: $GENERATED_OUTPUT_DIR
- Baseline JSONs: $BASELINE_OUTPUT_DIR
- Ablation JSONs: $ABLATION_OUTPUT_DIR
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models) MODELS="$2"; shift 2 ;;
    --benchmarks) BENCHMARKS="$2"; shift 2 ;;
    --strategies) STRATEGIES="$2"; shift 2 ;;
    --token-budgets) TOKEN_BUDGETS="$2"; shift 2 ;;
    --step-budgets) STEP_BUDGETS="$2"; shift 2 ;;
    --synthesis-iterations) SYNTH_ITERS="$2"; shift 2 ;;
    --generation-models) GEN_MODELS="$2"; shift 2 ;;
    --smiles-classes|--smiles-class) SMILES_CLASSES="$2"; shift 2 ;;
    --eval-backend) EVAL_BACKEND="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --eval-sample-size) EVAL_SAMPLE_SIZE="$2"; shift 2 ;;
    --eval-max-steps) EVAL_MAX_STEPS="$2"; shift 2 ;;
    --gsm-split-file) GSM_SPLIT_FILE="$2"; shift 2 ;;
    --spider-split-file) SPIDER_SPLIT_FILE="$2"; shift 2 ;;
    --vllm-gpu-memory-utilization) VLLM_GPU_MEM_UTIL="$2"; shift 2 ;;
    --dafny-path) DAFNY_PATH="$2"; shift 2 ;;
    --generated-output-dir) GENERATED_OUTPUT_DIR="$2"; shift 2 ;;
    --baseline-output-dir) BASELINE_OUTPUT_DIR="$2"; shift 2 ;;
    --ablation-output-dir) ABLATION_OUTPUT_DIR="$2"; shift 2 ;;
    --recompute-baselines) BASELINE_CACHE_MODE="refresh"; shift ;;
    --reuse-baselines) BASELINE_CACHE_MODE="reuse"; shift ;;
    --skip-main) SKIP_MAIN=1; shift ;;
    --skip-ablations) SKIP_ABLATIONS=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

case "$BASELINE_CACHE_MODE" in
  reuse|refresh) ;;
  *)
    echo "Invalid baseline cache mode: $BASELINE_CACHE_MODE (expected reuse or refresh)" >&2
    exit 2
    ;;
esac

mkdir -p "$GENERATED_OUTPUT_DIR" "$BASELINE_OUTPUT_DIR" "$ABLATION_OUTPUT_DIR"

IFS=',' read -r -a MODELS_ARR <<< "$MODELS"
IFS=',' read -r -a BENCHMARKS_ARR <<< "$BENCHMARKS"
IFS=',' read -r -a STRATEGIES_ARR <<< "$STRATEGIES"
IFS=',' read -r -a TOKEN_BUDGETS_ARR <<< "$TOKEN_BUDGETS"
IFS=',' read -r -a SYNTH_ITERS_ARR <<< "$SYNTH_ITERS"
IFS=',' read -r -a GEN_MODELS_ARR <<< "$GEN_MODELS"
IFS=',' read -r -a STEP_BUDGETS_ARR <<< "$STEP_BUDGETS"
IFS=',' read -r -a SMILES_CLASSES_ARR <<< "$SMILES_CLASSES"

trim() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "$s"
}

NORMALIZED_SMILES_CLASSES=()
seen_smiles_classes=""
for raw_smiles_class in "${SMILES_CLASSES_ARR[@]}"; do
  smiles_class="$(trim "$raw_smiles_class")"
  if [[ -z "$smiles_class" ]]; then
    continue
  fi
  case "$smiles_class" in
    acrylates|chain_extenders|isocyanates) ;;
    *)
      echo "Unknown SMILES class: $smiles_class" >&2
      echo "Expected one of: acrylates, chain_extenders, isocyanates" >&2
      exit 2
      ;;
  esac
  case ",$seen_smiles_classes," in
    *,"$smiles_class",*) continue ;;
  esac
  NORMALIZED_SMILES_CLASSES+=("$smiles_class")
  seen_smiles_classes="${seen_smiles_classes:+$seen_smiles_classes,}$smiles_class"
done

if [[ "${#NORMALIZED_SMILES_CLASSES[@]}" -eq 0 ]]; then
  echo "At least one SMILES class is required." >&2
  exit 2
fi

SMILES_CLASSES_ARR=("${NORMALIZED_SMILES_CLASSES[@]}")
SMILES_CLASSES="${SMILES_CLASSES_ARR[*]}"

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

# Resolve synthesis/generation profile labels used by metadecode ablations.
# gpt5.4 → OpenAI (GPT-5 class models not on Bedrock). opus4.7 → Bedrock Claude.
# gemini-pro: omitted from DEFAULT_GEN_MODELS; add --generation-models gemini-pro when wired.
resolve_gen_profile() {
  local profile="$1"
  local bedrock_model="${BEDROCK_GENERATION_MODEL:-${AWS_BEDROCK_GENERATION_MODEL:-anthropic.claude-3-5-sonnet-20241022-v2:0}}"
  local opus_model="${BEDROCK_OPUS_MODEL:-${BEDROCK_PROFILE_OPUS:-us.anthropic.claude-opus-4-1-20250514-v1:0}}"
  local openai_gpt="${OPENAI_GENERATION_MODEL:-gpt-5.4}"
  case "$profile" in
    gpt5.4)
      echo "openai|$openai_gpt"
      ;;
    opus4.7)
      echo "bedrock|$opus_model"
      ;;
    gemini-pro)
      echo "bedrock|${GEMINI_BEDROCK_MODEL:?Set GEMINI_BEDROCK_MODEL for gemini-pro (partner-owned wiring)}"
      ;;
    bedrock)
      echo "bedrock|$bedrock_model"
      ;;
    bedrock:*)
      echo "bedrock|${profile#bedrock:}"
      ;;
    *)
      echo "bedrock|$profile"
      ;;
  esac
}

_cuda_oom_detected_in_file() {
  local f="$1"
  [[ -f "$f" ]] || return 1
  # PyTorch / vLLM / CUDA runtime wording varies by version.
  grep -Eiq \
    'out of memory|OutOfMemoryError|CUDA out of memory|CUDA error: out of memory|torch\.cuda\.OutOfMemoryError|cumemAllocator|RESOURCE_EXHAUSTED' \
    "$f"
}

run_cmd() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] $*"
    return 0
  fi

  local fallback="${RUN_ALL_TESTS_CUDA_OOM_FALLBACK:-}"
  if [[ -z "$fallback" ]]; then
    echo "[run] $*"
    "$@"
    return $?
  fi

  local primary="${RUN_ALL_TESTS_CUDA_DEVICES:-3}"
  local log ec
  log="$(mktemp "${TMPDIR:-/tmp}/run_all_tests_cuda_try.XXXXXX")"

  echo "[run] CUDA_VISIBLE_DEVICES=$primary $*"
  (
    set -o pipefail
    CUDA_VISIBLE_DEVICES="$primary" "$@" 2>&1 | tee "$log"
  )
  ec=$?

  if [[ "$ec" -eq 0 ]]; then
    rm -f "$log"
    return 0
  fi

  if ! _cuda_oom_detected_in_file "$log"; then
    rm -f "$log"
    return "$ec"
  fi

  rm -f "$log"

  echo "[warn] CUDA OOM on CUDA_VISIBLE_DEVICES=$primary; retrying with CUDA_VISIBLE_DEVICES=$fallback" >&2
  echo "[run] CUDA_VISIBLE_DEVICES=$fallback $*"
  CUDA_VISIBLE_DEVICES="$fallback" "$@"
}

baseline_json_complete() {
  local path="$1"
  python - "$path" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    payload = json.loads(path.read_text())
except Exception:
    raise SystemExit(1)
answers = payload.get("answers")
if not isinstance(answers, list) or len(answers) == 0:
    raise SystemExit(1)
for row in answers:
    if not isinstance(row, dict):
        raise SystemExit(1)
    if "generated_answer" not in row:
        raise SystemExit(1)
raise SystemExit(0)
PY
}

baseline_json_matches_strategy() {
  local path="$1"
  local strategy="$2"
  if [[ "$strategy" != "crane" ]]; then
    return 0
  fi
  python - "$path" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    payload = json.loads(path.read_text())
except Exception:
    raise SystemExit(1)
if payload.get("metrics", {}).get("adapter") != "crane_shared_evaluator":
    raise SystemExit(1)
raise SystemExit(0)
PY
}

CSD_TARGET_STRATEGIES=(crane itergen cars)
declare -A PREPARED_BASELINES=()

baseline_case_key() {
  local strategy="$1"
  local model_slug="$2"
  local benchmark_key="$3"
  local token_budget="$4"
  local max_steps="$5"
  printf "%s|%s|%s|%s|%s" "$strategy" "$model_slug" "$benchmark_key" "$token_budget" "$max_steps"
}

baseline_json_usable() {
  local path="$1"
  local strategy="$2"
  [[ -f "$path" ]] \
    && [[ "$(wc -c < "$path")" -gt 20 ]] \
    && baseline_json_complete "$path" \
    && baseline_json_matches_strategy "$path" "$strategy"
}

best_csd_baseline_targets() {
  local benchmark="$1"
  local eval_model="$2"
  local token_budget="$3"
  local max_steps="$4"
  local smiles_class="${5:-}"

  local model_slug benchmark_key
  model_slug="$(slugify "$eval_model")"
  benchmark_key="$benchmark"
  if [[ "$benchmark" == "smiles" ]]; then
    benchmark_key="${benchmark}__class_$(slugify "$smiles_class")"
  fi

  python - "$BASELINE_OUTPUT_DIR" "$model_slug" "$benchmark_key" "$token_budget" "$max_steps" <<'PY'
import json
import sys
from pathlib import Path

baseline_dir = Path(sys.argv[1])
model_slug = sys.argv[2]
benchmark_key = sys.argv[3]
token_budget = sys.argv[4]
max_steps = sys.argv[5]

best_accuracy = None
best_syntax = None
for strategy in ("crane", "itergen", "cars"):
    path = (
        baseline_dir
        / strategy
        / model_slug
        / f"{benchmark_key}__tb{token_budget}__ms{max_steps}.json"
    )
    try:
        payload = json.loads(path.read_text())
    except Exception:
        continue

    answers = payload.get("answers")
    if not isinstance(answers, list) or not answers:
        continue
    if not all(isinstance(row, dict) and "generated_answer" in row for row in answers):
        continue
    if strategy == "crane" and payload.get("metrics", {}).get("adapter") != "crane_shared_evaluator":
        continue

    accuracy = payload.get("accuracy")
    if isinstance(accuracy, (int, float)):
        candidate = (float(accuracy), strategy, str(path), f"{float(accuracy):.1%}")
        if best_accuracy is None or candidate[0] > best_accuracy[0]:
            best_accuracy = candidate

    syntax_rate = payload.get("syntax_rate")
    if isinstance(syntax_rate, (int, float)):
        candidate = (float(syntax_rate), strategy, str(path), f"{float(syntax_rate):.1%}")
        if best_syntax is None or candidate[0] > best_syntax[0]:
            best_syntax = candidate

if best_accuracy is None:
    best_accuracy = (0.0, "none", "", "0.0%")
if best_syntax is None:
    best_syntax = (0.0, "none", "", "0.0%")

print(
    f"{best_accuracy[0]:.12g}|{best_accuracy[1]}|{best_accuracy[2]}|{best_accuracy[3]}|"
    f"{best_syntax[0]:.12g}|{best_syntax[1]}|{best_syntax[2]}|{best_syntax[3]}"
)
PY
}

ensure_csd_target_baselines() {
  local benchmark="$1"
  local eval_model="$2"
  local token_budget="$3"
  local max_steps="$4"
  local smiles_class="${5:-}"
  local strategy

  for strategy in "${CSD_TARGET_STRATEGIES[@]}"; do
    if ! run_fixed_strategy_case "$strategy" "$benchmark" "$eval_model" "$token_budget" "$max_steps" "$smiles_class"; then
      echo "[warn] Could not prepare $strategy baseline for benchmark=$benchmark eval_model=$eval_model token_budget=$token_budget max_steps=$max_steps smiles_class=${smiles_class:-<none>}" >&2
    fi
  done
}

run_fixed_strategy_case() {
  local strategy="$1"
  local benchmark="$2"
  local eval_model="$3"
  local token_budget="$4"
  local max_steps="$5"
  local smiles_class="${6:-}"

  local model_slug benchmark_key
  model_slug="$(slugify "$eval_model")"
  benchmark_key="$benchmark"
  if [[ "$benchmark" == "smiles" ]]; then
    if [[ -z "$smiles_class" ]]; then
      echo "Internal error: SMILES fixed-strategy run requires a class." >&2
      return 2
    fi
    benchmark_key="${benchmark}__class_$(slugify "$smiles_class")"
  fi

  local out_json="${BASELINE_OUTPUT_DIR}/${strategy}/${model_slug}/${benchmark_key}__tb${token_budget}__ms${max_steps}.json"
  mkdir -p "$(dirname "$out_json")"
  local case_key
  case_key="$(baseline_case_key "$strategy" "$model_slug" "$benchmark_key" "$token_budget" "$max_steps")"
  local allow_cache_reuse=0
  if [[ "$BASELINE_CACHE_MODE" == "reuse" || -n "${PREPARED_BASELINES[$case_key]+set}" ]]; then
    allow_cache_reuse=1
  fi

  if [[ "$allow_cache_reuse" -eq 1 ]] && baseline_json_usable "$out_json" "$strategy"; then
    PREPARED_BASELINES["$case_key"]=1
    echo "[skip] $out_json already exists ($(wc -l < "$out_json") lines). Delete it to re-run."
    return 0
  elif [[ -f "$out_json" ]]; then
    if [[ "$BASELINE_CACHE_MODE" == "refresh" && -z "${PREPARED_BASELINES[$case_key]+set}" ]]; then
      echo "[rerun] $out_json exists but --recompute-baselines was requested."
    else
      echo "[rerun] $out_json exists but is incomplete, corrupt, or from an obsolete adapter."
    fi
  fi

  # All fixed strategies (unconstrained, gcd, crane, itergen, cars) use legacy adapters.
  local cmd=(
    python -m synthesis.evaluate.run_legacy_fixed_strategy
    --strategy "$strategy"
    --dataset "$benchmark"
    --eval-model "$eval_model"
    --eval-backend "$EVAL_BACKEND"
    --device "$DEVICE"
    --eval-sample-size "$EVAL_SAMPLE_SIZE"
    --eval-max-steps "$max_steps"
    --eval-step-token-budget "$token_budget"
    --vllm-gpu-memory-utilization "$VLLM_GPU_MEM_UTIL"
    --output-json "$out_json"
  )

  if [[ -n "$DAFNY_PATH" ]]; then
    cmd+=(--dafny-path "$DAFNY_PATH")
  fi

  if [[ -n "$GSM_SPLIT_FILE" ]] && [[ "$benchmark" == "gsm_symbolic" ]]; then
    cmd+=(--gsm-split-file "$GSM_SPLIT_FILE" --gsm-split-name eval)
  fi
  if [[ -n "$SPIDER_SPLIT_FILE" ]] && [[ "$benchmark" == "spider" ]]; then
    cmd+=(--spider-split-file "$SPIDER_SPLIT_FILE" --spider-split-name eval)
  fi
  if [[ "$benchmark" == "smiles" ]]; then
    cmd+=(--smiles-classes "$smiles_class" --smiles-samples-per-class "$EVAL_SAMPLE_SIZE")
  fi

  if run_cmd "${cmd[@]}"; then
    PREPARED_BASELINES["$case_key"]=1
    return 0
  fi
  return 1
}

run_fixed_strategy_cases() {
  local strategy="$1"
  local benchmark="$2"
  local eval_model="$3"
  local token_budget="$4"
  local max_steps="$5"

  if [[ "$benchmark" == "smiles" ]]; then
    local smiles_class
    for smiles_class in "${SMILES_CLASSES_ARR[@]}"; do
      run_fixed_strategy_case "$strategy" "$benchmark" "$eval_model" "$token_budget" "$max_steps" "$smiles_class"
    done
    return 0
  fi

  run_fixed_strategy_case "$strategy" "$benchmark" "$eval_model" "$token_budget" "$max_steps"
}

run_metadecode_case() {
  local benchmark="$1"
  local eval_model="$2"
  local token_budget="$3"
  local synth_iter="$4"
  local gen_profile="$5"
  local max_steps="$6"
  local smiles_class="${7:-}"

  local resolved backend generation_model
  resolved="$(resolve_gen_profile "$gen_profile")"
  backend="${resolved%%|*}"
  generation_model="${resolved##*|}"

  local model_slug gen_slug run_name task class_suffix benchmark_key
  model_slug="$(slugify "$eval_model")"
  gen_slug="$(slugify "$gen_profile")"
  class_suffix=""
  benchmark_key="$benchmark"
  if [[ "$benchmark" == "smiles" ]]; then
    if [[ -z "$smiles_class" ]]; then
      echo "Internal error: SMILES metadecode run requires a class." >&2
      return 2
    fi
    class_suffix="_class_$(slugify "$smiles_class")"
    benchmark_key="${benchmark}__class_$(slugify "$smiles_class")"
  fi
  run_name="metadecode_${benchmark}_${model_slug}_${gen_slug}_iter${synth_iter}_tb${token_budget}_ms${max_steps}${class_suffix}"
  task="$(metadecode_task "$benchmark")"
  ensure_csd_target_baselines "$benchmark" "$eval_model" "$token_budget" "$max_steps" "$smiles_class"
  local target_accuracy target_strategy target_path target_percent
  local target_syntax target_syntax_strategy target_syntax_path target_syntax_percent
  IFS='|' read -r target_accuracy target_strategy target_path target_percent target_syntax target_syntax_strategy target_syntax_path target_syntax_percent < <(
    best_csd_baseline_targets "$benchmark" "$eval_model" "$token_budget" "$max_steps" "$smiles_class"
  )
  if [[ "$target_strategy" == "none" && "$target_syntax_strategy" == "none" ]]; then
    echo "[target] metadecode ${benchmark_key}/${model_slug} tb${token_budget} ms${max_steps}: no valid CRANE/IterGen/CARS baseline found; passing --min-accuracy 0.0 --min-syntax-rate 0.0"
  else
    echo "[target] metadecode ${benchmark_key}/${model_slug} tb${token_budget} ms${max_steps}: best CSD baseline accuracy ${target_strategy}=${target_percent}, syntax ${target_syntax_strategy}=${target_syntax_percent}; passing --min-accuracy ${target_accuracy} --min-syntax-rate ${target_syntax}"
  fi

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
    --min-accuracy "$target_accuracy"
    --min-syntax-rate "$target_syntax"
    --eval-sample-size "$EVAL_SAMPLE_SIZE"
    --eval-max-steps "$max_steps"
    --eval-step-token-budget "$token_budget"
    --vllm-gpu-memory-utilization "$VLLM_GPU_MEM_UTIL"
    --device "$DEVICE"
    --output-dir "$GENERATED_OUTPUT_DIR"
  )

  if [[ -n "$GSM_SPLIT_FILE" ]] && [[ "$benchmark" == "gsm_symbolic" ]]; then
    synth_cmd+=(--gsm-split-file "$GSM_SPLIT_FILE" --gsm-split-name eval)
  fi
  if [[ -n "$SPIDER_SPLIT_FILE" ]] && [[ "$benchmark" == "spider" ]]; then
    synth_cmd+=(--spider-split-file "$SPIDER_SPLIT_FILE" --spider-split-name eval)
  fi
  if [[ "$benchmark" == "smiles" ]]; then
    synth_cmd+=(--smiles-samples-per-class "$EVAL_SAMPLE_SIZE" --smiles-classes "$smiles_class")
  fi

  if [[ -n "$DAFNY_PATH" ]]; then
    synth_cmd+=(--dafny-path "$DAFNY_PATH")
  fi

  if ! run_cmd "${synth_cmd[@]}"; then
    echo "[warn] Metadecode synthesis failed for benchmark=$benchmark eval_model=$eval_model token_budget=$token_budget iter=$synth_iter gen=$gen_profile max_steps=$max_steps" >&2
    return 0
  fi

  local out_json
  out_json="${BASELINE_OUTPUT_DIR}/metadecode/${model_slug}/${benchmark_key}__tb${token_budget}__ms${max_steps}__gen${gen_slug}__iter${synth_iter}.json"
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

run_metadecode_cases() {
  local benchmark="$1"
  local eval_model="$2"
  local token_budget="$3"
  local synth_iter="$4"
  local gen_profile="$5"
  local max_steps="$6"

  if [[ "$benchmark" == "smiles" ]]; then
    local smiles_class
    for smiles_class in "${SMILES_CLASSES_ARR[@]}"; do
      run_metadecode_case "$benchmark" "$eval_model" "$token_budget" "$synth_iter" "$gen_profile" "$max_steps" "$smiles_class"
    done
    return 0
  fi

  run_metadecode_case "$benchmark" "$eval_model" "$token_budget" "$synth_iter" "$gen_profile" "$max_steps"
}

echo "=== run_all_tests matrix ==="
echo "models: ${MODELS_ARR[*]}"
echo "benchmarks: ${BENCHMARKS_ARR[*]}"
echo "strategies: ${STRATEGIES_ARR[*]}"
echo "token budgets: ${TOKEN_BUDGETS_ARR[*]}"
echo "step budgets (ablation): ${STEP_BUDGETS_ARR[*]}"
echo "synthesis iters (metadecode): ${SYNTH_ITERS_ARR[*]}"
echo "generation models (metadecode): ${GEN_MODELS_ARR[*]}"
echo "SMILES classes: ${SMILES_CLASSES_ARR[*]}"
echo "eval max steps (main): ${EVAL_MAX_STEPS}"
echo "baseline cache mode: ${BASELINE_CACHE_MODE} (reuse=skip complete JSONs, refresh=recompute fixed baselines)"
echo ""

# ========================================================
# Phase 1: Main matrix (model x benchmark x strategy)
# Uses EVAL_MAX_STEPS for all runs, first token budget only
# ========================================================
if [[ "$SKIP_MAIN" -eq 0 ]]; then
  echo "=== Phase 1: Main experiment matrix ==="
  for eval_model in "${MODELS_ARR[@]}"; do
    for raw_benchmark in "${BENCHMARKS_ARR[@]}"; do
      benchmark="$(normalize_benchmark "$raw_benchmark")"
      for strategy in "${STRATEGIES_ARR[@]}"; do
        if [[ "$strategy" == "metadecode" ]]; then
          run_metadecode_cases "$benchmark" "$eval_model" "${TOKEN_BUDGETS_ARR[0]}" \
            "${SYNTH_ITERS_ARR[-1]}" "${GEN_MODELS_ARR[0]}" "$EVAL_MAX_STEPS"
        else
          run_fixed_strategy_cases "$strategy" "$benchmark" "$eval_model" \
            "${TOKEN_BUDGETS_ARR[0]}" "$EVAL_MAX_STEPS"
        fi
      done
    done
  done
  echo "=== Phase 1 complete ==="
fi

# ========================================================
# Phase 2: Ablation studies
# ========================================================
if [[ "$SKIP_ABLATIONS" -eq 0 ]]; then
  echo ""
  echo "=== Phase 2: Ablation studies ==="

  # Default ablation model (7B)
  ABLATION_MODEL="Qwen/Qwen2.5-Coder-7B-Instruct"

  # --- Ablation A: Step budget (max_steps) ---
  echo "--- Ablation A: Step budget ---"
  for raw_benchmark in "gsm" "spider" "smiles"; do
    benchmark="$(normalize_benchmark "$raw_benchmark")"
    for step_budget in "${STEP_BUDGETS_ARR[@]}"; do
      for strategy in "gcd" "crane" "itergen" "cars" "metadecode"; do
        if [[ "$strategy" == "metadecode" ]]; then
          run_metadecode_cases "$benchmark" "$ABLATION_MODEL" "${TOKEN_BUDGETS_ARR[0]}" \
            "${SYNTH_ITERS_ARR[-1]}" "${GEN_MODELS_ARR[0]}" "$step_budget"
        else
          run_fixed_strategy_cases "$strategy" "$benchmark" "$ABLATION_MODEL" \
            "${TOKEN_BUDGETS_ARR[0]}" "$step_budget"
        fi
      done
    done
  done

  # --- Ablation B: Synthesis iterations K ---
  echo "--- Ablation B: Synthesis iterations ---"
  for raw_benchmark in "gsm" "spider" "smiles"; do
    benchmark="$(normalize_benchmark "$raw_benchmark")"
    for synth_iter in "${SYNTH_ITERS_ARR[@]}"; do
      run_metadecode_cases "$benchmark" "$ABLATION_MODEL" "${TOKEN_BUDGETS_ARR[0]}" \
        "$synth_iter" "${GEN_MODELS_ARR[0]}" "$EVAL_MAX_STEPS"
    done
  done

  # --- Ablation C: Synthesizer model ---
  echo "--- Ablation C: Synthesizer model ---"
  for raw_benchmark in "gsm" "spider" "smiles"; do
    benchmark="$(normalize_benchmark "$raw_benchmark")"
    for gen_profile in "${GEN_MODELS_ARR[@]}"; do
      run_metadecode_cases "$benchmark" "$ABLATION_MODEL" "${TOKEN_BUDGETS_ARR[0]}" \
        "${SYNTH_ITERS_ARR[-1]}" "$gen_profile" "$EVAL_MAX_STEPS"
    done
  done

  # --- Ablation D: Per-step token budget ---
  echo "--- Ablation D: Per-step token budget ---"
  for raw_benchmark in "gsm" "spider" "smiles"; do
    benchmark="$(normalize_benchmark "$raw_benchmark")"
    for token_budget in "${TOKEN_BUDGETS_ARR[@]}"; do
      for strategy in "gcd" "crane" "itergen" "cars" "metadecode"; do
        if [[ "$strategy" == "metadecode" ]]; then
          run_metadecode_cases "$benchmark" "$ABLATION_MODEL" "$token_budget" \
            "${SYNTH_ITERS_ARR[-1]}" "${GEN_MODELS_ARR[0]}" "$EVAL_MAX_STEPS"
        else
          run_fixed_strategy_cases "$strategy" "$benchmark" "$ABLATION_MODEL" \
            "$token_budget" "$EVAL_MAX_STEPS"
        fi
      done
    done
  done

  # --- Ablation E: Beam refinement x adaptive helper masking x helper selection policy ---
  echo "--- Ablation E: Beam refinement x adaptive helper masking x helper selection policy ---"
  for raw_benchmark in "gsm" "spider" "smiles"; do
    benchmark="$(normalize_benchmark "$raw_benchmark")"
    for beam_size in 1 2 4; do
      for mask_flag in "--adaptive-helper-mask" "--no-adaptive-helper-mask"; do
        mask_label="mask_on"
        if [[ "$mask_flag" == "--no-adaptive-helper-mask" ]]; then
          mask_label="mask_off"
        fi
        for policy in "utility" "bandit"; do
          ablation_smiles_classes=("")
          if [[ "$benchmark" == "smiles" ]]; then
            ablation_smiles_classes=("${SMILES_CLASSES_ARR[@]}")
          fi
          for smiles_class in "${ablation_smiles_classes[@]}"; do
            task_e="$(metadecode_task "$benchmark")"
            class_suffix=""
            if [[ "$benchmark" == "smiles" ]]; then
              class_suffix="_class_$(slugify "$smiles_class")"
            fi
            run_name_e="ablat_beam${beam_size}_${mask_label}_${policy}_${benchmark}${class_suffix}"
            resolved_gen_e="$(resolve_gen_profile "gpt5.4")"
            backend_e="${resolved_gen_e%%|*}"
            generation_model_e="${resolved_gen_e##*|}"
            ensure_csd_target_baselines "$benchmark" "$ABLATION_MODEL" "${TOKEN_BUDGETS_ARR[0]}" "$EVAL_MAX_STEPS" "$smiles_class"
            target_accuracy_e="0.0"
            target_strategy_e="none"
            target_percent_e="0.0%"
            target_syntax_e="0.0"
            target_syntax_strategy_e="none"
            target_syntax_percent_e="0.0%"
            IFS='|' read -r target_accuracy_e target_strategy_e target_path_e target_percent_e target_syntax_e target_syntax_strategy_e target_syntax_path_e target_syntax_percent_e < <(
              best_csd_baseline_targets "$benchmark" "$ABLATION_MODEL" "${TOKEN_BUDGETS_ARR[0]}" "$EVAL_MAX_STEPS" "$smiles_class"
            )
            if [[ "$target_strategy_e" == "none" && "$target_syntax_strategy_e" == "none" ]]; then
              echo "[target] metadecode ${benchmark}${class_suffix}/$(slugify "$ABLATION_MODEL") tb${TOKEN_BUDGETS_ARR[0]} ms${EVAL_MAX_STEPS}: no valid CRANE/IterGen/CARS baseline found; passing --min-accuracy 0.0 --min-syntax-rate 0.0"
            else
              echo "[target] metadecode ${benchmark}${class_suffix}/$(slugify "$ABLATION_MODEL") tb${TOKEN_BUDGETS_ARR[0]} ms${EVAL_MAX_STEPS}: best CSD baseline accuracy ${target_strategy_e}=${target_percent_e}, syntax ${target_syntax_strategy_e}=${target_syntax_percent_e}; passing --min-accuracy ${target_accuracy_e} --min-syntax-rate ${target_syntax_e}"
            fi
            cmd_e=(
              python -m synthesis.run_synthesis
              --task "$task_e"
              --dataset "$benchmark"
              --generation-backend "$backend_e"
              --generation-model "$generation_model_e"
              --eval-model "$ABLATION_MODEL"
              --eval-backend "$EVAL_BACKEND"
              --max-iterations "${SYNTH_ITERS_ARR[-1]}"
              --output-name "$run_name_e"
              --min-accuracy "$target_accuracy_e"
              --min-syntax-rate "$target_syntax_e"
              --eval-sample-size "$EVAL_SAMPLE_SIZE"
              --eval-max-steps "$EVAL_MAX_STEPS"
              --eval-step-token-budget "${TOKEN_BUDGETS_ARR[0]}"
              --vllm-gpu-memory-utilization "$VLLM_GPU_MEM_UTIL"
              --device "$DEVICE"
              --output-dir "$GENERATED_OUTPUT_DIR"
              --refinement-beam-size "$beam_size"
              "$mask_flag"
              --helper-selection-policy "$policy"
            )
            if [[ -n "$GSM_SPLIT_FILE" ]] && [[ "$benchmark" == "gsm_symbolic" ]]; then
              cmd_e+=(--gsm-split-file "$GSM_SPLIT_FILE" --gsm-split-name eval)
            fi
            if [[ -n "$SPIDER_SPLIT_FILE" ]] && [[ "$benchmark" == "spider" ]]; then
              cmd_e+=(--spider-split-file "$SPIDER_SPLIT_FILE" --spider-split-name eval)
            fi
            if [[ "$benchmark" == "smiles" ]]; then
              cmd_e+=(--smiles-samples-per-class "$EVAL_SAMPLE_SIZE" --smiles-classes "$smiles_class")
            fi
            if [[ -n "$DAFNY_PATH" ]]; then
              cmd_e+=(--dafny-path "$DAFNY_PATH")
            fi
            run_cmd "${cmd_e[@]}"
          done
        done
      done
    done
  done

  echo "=== Phase 2 complete ==="
fi

echo ""
echo "All requested matrix jobs completed."
