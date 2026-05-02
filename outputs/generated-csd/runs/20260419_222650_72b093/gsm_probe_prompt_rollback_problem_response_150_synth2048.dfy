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
    // CSD_RATIONALE_BEGIN
// This strategy preserves valid constrained spans while generating GSM-symbolic style
// reasoning. Outside a span, it samples mostly unconstrained text but lightly boosts
// the opening delimiter token "<<" so symbolic names can be introduced. Inside a
// span, it uses parser-guided constrained decoding only over valid continuations,
// with stricter decoding when the valid continuation set is narrow. A constrained
// span is closed only in a branch already guarded by parser.IsCompletePrefix on the
// current constrained contents. EOS is always terminal. To avoid unverifiable token-
// membership preconditions, the strategy does not use helper calls that require
// proving ad hoc token lists are in lm.Tokens.
// CSD_RATIONALE_END
generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

var steps := 0;

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
    lm.GenerateLogits(prompt + generated);
    helpers.BoostTokenLogits(lm, ["<<"], 2.0);
    if |generated| > 0 {
      var lastTok := generated[|generated| - 1];
      if Contains(lastTok, "=") ||
         Contains(lastTok, "answer") || Contains(lastTok, "Answer") ||
         Contains(lastTok, "final") || Contains(lastTok, "Final") ||
         Contains(lastTok, "quantity") || Contains(lastTok, "Quantity") ||
         Contains(lastTok, "symbol") || Contains(lastTok, "Symbol") {
        helpers.BoostTokenLogits(lm, ["<<"], 8.0);
      }
    }
    var next := lm.ChooseNextTokenUnconstrained();
    helpers.cost := helpers.cost + 1;
    steps := steps + 1;
    if next == eosToken {
      // EOS is terminal
    } else {
      generated := generated + [next];
      if next == "<<" {
        insideConstrainedOut := true;
        currentConstrainedOut := [];
      }
    }
    if next == eosToken {
      steps := maxSteps;
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
      var stablePrefix := generated[..|generated| - |currentConstrainedOut|];
      var constrainedPrompt := prompt + stablePrefix;
      var validCount := helpers.ValidTokenCount(parser, currentConstrainedOut);
      if validCount <= 2 {
        var next := helpers.ConstrainedStep(lm, parser, constrainedPrompt, currentConstrainedOut, eosToken);
        steps := steps + 1;
        if next == eosToken {
          steps := maxSteps;
        } else {
          var appendedGenerated, appendedInside, appendedCurrent := helpers.AppendConstrainedToken(
            lm, parser, generated, currentConstrainedOut, next
          );
          generated := appendedGenerated;
          insideConstrainedOut := appendedInside;
          currentConstrainedOut := appendedCurrent;
        }
      } else {
        var next, isValid := helpers.SoftConstrainedStep(
          lm, parser, constrainedPrompt, currentConstrainedOut, 3.0, eosToken
        );
        steps := steps + 1;
        if next == eosToken {
          steps := maxSteps;
        } else {
          if isValid {
            var appendedGenerated2, appendedInside2, appendedCurrent2 := helpers.AppendConstrainedToken(
              lm, parser, generated, currentConstrainedOut, next
            );
            generated := appendedGenerated2;
            insideConstrainedOut := appendedInside2;
            currentConstrainedOut := appendedCurrent2;
          } else {
            var repairedGenerated, repairedCurrent := helpers.RollbackConstrainedSpan(
              parser, stablePrefix, generated, currentConstrainedOut
            );
            generated := repairedGenerated;
            currentConstrainedOut := repairedCurrent;
            insideConstrainedOut := true;
          }
        }
      }
    }
  }
}

cost := steps;
  }
}
