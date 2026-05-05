#!/bin/bash
# Generate a FOLIO-specific CSD for first-order logic reasoning
#
# Usage: bash scripts/generate_folio_csd.sh

set -e

if [[ -f ".env" ]]; then
  set -a
  source .env
  set +a
fi

DAFNY_PATH_ARG=("--dafny-path" "${DAFNY_PATH:-/home/aadivyar/.dotnet/tools/dafny}")

# FOLIO task description for FOL constrained decoding
TASK_DESC="Generate first-order logic (FOL) formulas for FOLIO logical reasoning. \
The parser enforces a strict FOL grammar with quantifiers, predicates, and logical connectives. \
CRITICAL RULES: \
1. Use {forall} and {exists} for quantifiers, followed by a single lowercase variable and a formula. \
2. Predicates start with uppercase (e.g., Dog(x), Likes(john, mary)) with lowercase arguments. \
3. Single lowercase letters (x, y, z) are variables; multi-character lowercase identifiers are constants. \
4. Logical connectives: {and}, {or}, {not}, {implies}, {iff}, {xor}. \
5. The constrained windows contain complete FOL statements — one per premise and one for the conclusion. \
6. Parentheses are used for grouping; operator precedence is: iff < implies < xor < or < and < not."

echo "Generating FOLIO-specific CSD for FOL reasoning..."
echo ""
echo "Task description:"
echo "  $TASK_DESC"
echo ""

# Run synthesis
python run_synthesis.py \
    --task "$TASK_DESC" \
    --dataset folio \
    --cost-contract "" \
    --max-iterations 10 \
    --generation-model "gpt-5.4" \
    --generation-backend openai \
    --eval-model "Qwen/Qwen2.5-Coder-7B-Instruct" \
    --eval-backend vllm \
    --output-name "folio_csd" \
    --temperature 0.7 \
    "${DAFNY_PATH_ARG[@]}" \
    --device auto \
    --min-accuracy 0.5 \
    --min-syntax-rate 0.8 \
    --eval-sample-size 10 \
    --synthesis-max-tokens 3072

echo ""
echo "FOLIO CSD generation complete!"
echo ""
echo "To use the generated CSD, run:"
echo "python -m evaluations.folio.cli \\"
echo "   --run-dir outputs/generated-csd/runs/latest \\"
echo "   --model Qwen/Qwen2.5-Coder-7B-Instruct \\"
echo "   --device cuda \\"
echo "   --limit 10 \\"
echo "   --max-steps 1500 \\"
echo "   --vocab-size 2000 \\"
echo "   --verbose"
