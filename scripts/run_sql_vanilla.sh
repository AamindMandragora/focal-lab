#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

if [[ -f ".env" ]]; then
  set -a
  source .env
  set +a
fi

DAFNY_PATH_ARG=("--dafny-path" "${DAFNY_PATH:-/home/aadivyar/.dotnet/tools/dafny}")

# Script to evaluate Spider text-to-SQL using the vanilla CRANE system
# (no synthesized CSD — just PureConstrainedGeneration on the SQL grammar).

echo "Setting up vanilla CRANE strategy in dafny/GeneratedCSD.dfy..."

cat > dafny/GeneratedCSD.dfy <<EOF
include "VerifiedAgentSynthesis.dfy"

module GeneratedCSD {
  import opened VerifiedDecoderAgent

  method MyCSDStrategy(lm: LM, parser: Parser, prompt: Prefix, maxSteps: nat) returns (generated: Prefix)
    modifies lm.Logits
    requires lm.ValidTokensIdsLogits()
    requires parser.IsValidPrefix([])
    requires forall t: Token :: t in parser.ValidNextTokens([]) ==> t in lm.Tokens
    ensures lm.ValidTokensIdsLogits()
    ensures |generated| <= maxSteps
    ensures parser.IsValidPrefix(generated)
    ensures |generated| == maxSteps || parser.IsCompletePrefix(generated)
  {
    generated := CSDHelpers.PureConstrainedGeneration(lm, parser, prompt, maxSteps);
  }
}
EOF

# Compile the strategy (skip generation)
echo "Compiling strategy..."
python run_synthesis.py --task "Vanilla CRANE" --compile-only --output-name generated_csd --dataset spider "${DAFNY_PATH_ARG[@]}" > /dev/null

# Get the run directory
RUN_DIR=$(cat outputs/generated-csd/latest_run.txt)
echo "Using compiled run: $RUN_DIR"

# Run the evaluation (pass through additional args)
echo "Starting evaluation..."
echo "Arguments: $@"

python -m evaluations.sql_spider.cli \
  --run-dir "$RUN_DIR" \
  "$@"
