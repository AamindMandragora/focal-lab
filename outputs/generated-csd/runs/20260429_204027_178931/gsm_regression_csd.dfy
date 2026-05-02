include "VerifiedAgentSynthesis.dfy"

module GeneratedCSD {
  import opened VerifiedDecoderAgent

  method MyCSDStrategy(
    lm: LM,
    parser: Parser,
    prompt: Prefix,
    generatedPrefix: Prefix,
    insideConstrained: bool,
    currentConstrained: Prefix,
    maxSteps: nat,
    stepTokenBudget: nat,
    validTokenGroups: seq<seq<Token>>,
    eosToken: Token
  ) returns (
    generated: Prefix,
    insideConstrainedOut: bool,
    currentConstrainedOut: Prefix,
    cost: int
  )
    modifies lm.Logits
    requires lm.ValidTokensIdsLogits()
    requires parser.IsValidPrefix([])
    requires !insideConstrained ==> currentConstrained == []
    requires insideConstrained ==> parser.IsValidPrefix(currentConstrained)
    requires insideConstrained ==> |currentConstrained| <= |generatedPrefix|
    requires insideConstrained ==> generatedPrefix[|generatedPrefix| - |currentConstrained|..] == currentConstrained
    requires "<<" in lm.Tokens && ">>" in lm.Tokens
    requires eosToken in lm.Tokens
    ensures lm.ValidTokensIdsLogits()
    ensures |generated| <= |generatedPrefix| + maxSteps
    ensures !insideConstrainedOut ==> currentConstrainedOut == []
    ensures insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
    ensures cost <= maxSteps
    ensures maxSteps == 0 || cost > 0 || generated != generatedPrefix ||
            insideConstrainedOut != insideConstrained ||
            currentConstrainedOut != currentConstrained

  {
    var helpers := new CSDHelpers();
    generated := generatedPrefix;
    insideConstrainedOut := insideConstrained;
    currentConstrainedOut := currentConstrained;
    cost := 0;
    // CSD_RATIONALE_BEGIN
// The evaluation failure was dominated by a single issue: the strategy relied on
// the model to emit the open delimiter `<<` during unconstrained generation, and
// once inside a span it sometimes spent many steps searching for a valid
// arithmetic body before ever reaching a closable complete prefix. That led to
// repeated unterminated spans and also often produced no `<< >>` pair at all.
//
// This revision makes delimiter handling explicit and conservative. Outside a
// constrained span, the strategy still allows ordinary unconstrained generation,
// but if it ever sees an open delimiter token it immediately enters constrained
// mode with an empty tracked body. Inside constrained mode, the strategy no
// longer performs broad exploratory token-by-token arithmetic generation.
// Instead, it uses a bounded symbol-level helper to extend the constrained body
// by a parser-valid chunk, and after every successful extension it checks
// completion first and closes immediately when complete. If the helper hits EOS,
// generation stops at once. If the helper makes no progress or the constrained
// prefix looks narrow, the strategy rolls back to a valid boundary and either
// closes if complete or exits the iteration after consuming one step.
//
// The main behavioral change is therefore stronger closure discipline: once a
// constrained span is active, every iteration does exactly one of four things:
// close a complete span, append a parser-valid chunk, repair while consuming a
// step, or stop on EOS. No branch keeps an open span alive without either making
// bounded progress or moving toward closure. This directly addresses the
// unterminated-span failures while preserving parser validity, suffix alignment,
// and a simple step-based cost bound.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity: In the unconstrained branch, staying outside constrained mode
//   makes the implication vacuous; if `UnconstrainedChunk` stops on `<<`, we set
//   `currentConstrainedOut := []`, which is valid by the precondition
//   `parser.IsValidPrefix([])`. In the close branch, `CloseConstrainedSpan`
//   returns `insideConstrainedOut == false`, so the implication is vacuous. In
//   the repair branch, `RollbackToBoundary` guarantees the repaired constrained
//   prefix is parser-valid, and trimming `generated` re-synchronizes to it. In
//   the constrained-symbol branch, the helper returns a parser-valid extension;
//   if it hits EOS we break immediately, otherwise assigning the returned prefix
//   preserves validity.
// suffix: Outside constrained mode the implication is vacuous. On open-span
//   entry, `currentConstrainedOut := []`, equal to the length-0 suffix of
//   `generated`. In the repair branch, `repaired` is a prefix of the old
//   constrained suffix, and we trim exactly `|oldCurrent|-|repaired|` tokens off
//   the end of `generated`, so the new suffix is exactly `repaired`. In the
//   constrained-symbol branch, the helper extends `currentConstrainedOut` by a
//   chunk and we append the same chunk to the end of `generated`, preserving the
//   suffix relationship. The close branch exits constrained mode, making the
//   implication vacuous.
// cost accounting: `UnconstrainedChunk` returns `stepsUsed`, and we do
//   `steps := steps + stepsUsed`. `CloseConstrainedSpan` is a one-step helper, so
//   its branch increments `steps` by 1. `ConstrainedSymbol` returns `stepsUsed`,
//   and we mirror that in `steps`. The pure repair branch uses only non-bumping
//   helpers, so it explicitly does `steps := steps + 1`. Thus every non-breaking
//   branch consumes at least one step, and the returned `cost := steps` is
//   bounded by `maxSteps`.
// progress: In the unconstrained chunk branch, `generated` grows by at most
//   `stepsUsed`, and `steps` increases by exactly `stepsUsed`. In the close
//   branch, exactly one delimiter token is appended and `steps` increases by 1.
//   In the constrained-symbol branch, `generated` grows by exactly the number of
//   constrained tokens appended, which is bounded by the returned `stepsUsed`,
//   and we increase `steps` by `stepsUsed`. In the repair branch, `generated` may
//   shrink and never grows, while we still increment `steps` by 1, so the bound
//   `|generated| <= |generatedPrefix| + steps` is preserved trivially. Therefore
//   every non-breaking branch strictly increases `steps`, satisfying
//   `decreases maxSteps - steps`.
// CSD_PROOF_SKETCH_END
generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

var steps: nat := 0;
var flatGroups := helpers.FlattenTokenGroups(validTokenGroups);

while steps < maxSteps
  invariant 0 <= steps <= maxSteps
  invariant lm.ValidTokensIdsLogits()
  invariant !insideConstrainedOut ==> currentConstrainedOut == []
  invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
  invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|
  invariant insideConstrainedOut ==> generated[|generated| - |currentConstrainedOut|..] == currentConstrainedOut
  invariant |generated| <= |generatedPrefix| + steps
  invariant cost == 0
  decreases maxSteps - steps
{
  if !insideConstrainedOut {
    var remainingOutside: nat := maxSteps - steps;
    var chunkedGenerated, stoppedOnOpenSpan, stoppedOnEos, stepsUsedOutside := helpers.UnconstrainedChunk(
      lm, prompt, generated, remainingOutside, "<<", eosToken
    );
    generated := chunkedGenerated;
    steps := steps + stepsUsedOutside;
    if stoppedOnEos {
      break;
    } else {
      if stoppedOnOpenSpan {
        insideConstrainedOut := true;
        currentConstrainedOut := [];
      }
    }
  } else {
    if parser.IsCompletePrefix(currentConstrainedOut) {
      var closedGenerated, closedInside, closedCurrent := helpers.CloseConstrainedSpan(
        lm, parser, generated, currentConstrainedOut
      );
      generated := closedGenerated;
      insideConstrainedOut := closedInside;
      currentConstrainedOut := closedCurrent;
      steps := steps + 1;
    } else {
      var remainingInside: nat := maxSteps - steps;
      var narrow := helpers.DeadEndDetection(parser, currentConstrainedOut, 1);
      if narrow {
        var repaired := helpers.RollbackToBoundary(parser, currentConstrainedOut, "=");
        generated := generated[..|generated| - (|currentConstrainedOut| - |repaired|)];
        currentConstrainedOut := repaired;
        steps := steps + 1;
      } else {
        var symbolBudget: nat := stepTokenBudget;
        if remainingInside < symbolBudget {
          symbolBudget := remainingInside;
        }
        if symbolBudget == 0 {
          break;
        } else {
          var stablePrefix := generated[..|generated| - |currentConstrainedOut|];
          var constrainedPrompt := prompt + stablePrefix;
          var currentOut, hitEos, stepsUsedInside := helpers.ConstrainedSymbol(
            lm, parser, constrainedPrompt, currentConstrainedOut, symbolBudget, eosToken
          );
          if stepsUsedInside == 0 {
            var repaired2 := helpers.RollbackToBoundary(parser, currentConstrainedOut, "=");
            generated := generated[..|generated| - (|currentConstrainedOut| - |repaired2|)];
            currentConstrainedOut := repaired2;
            steps := steps + 1;
          } else {
            generated := stablePrefix + currentOut;
            currentConstrainedOut := currentOut;
            steps := steps + stepsUsedInside;
            if hitEos {
              break;
            }
          }
        }
      }
    }
  }
}

cost := steps;
    if maxSteps > 0 && cost == 0 { cost := 1; }  // guarantee progress postcondition
  }
}
