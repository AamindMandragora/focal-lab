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
// This strategy preserves valid constrained << >> spans while encouraging
// symbolic quantity-name generation for GSM-style math reasoning. Outside a
// constrained span, it decodes unconstrainedly but boosts the opening delimiter
// after common symbolic-definition or answer-cue tokens. Inside a constrained
// span, it uses parser-guided constrained decoding only, avoiding extra token
// assumptions that require membership proofs. When the constrained prefix is
// complete, it closes the span immediately from a branch explicitly guarded by
// parser completeness. If constrained decoding ever returns EOS, generation
// stops immediately. The active constrained suffix is always tracked separately
// from the full generated text.
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
    if |generated| > 0 {
      var lastTok := generated[|generated| - 1];
      if Contains(lastTok, "=") || Contains(lastTok, "let") || Contains(lastTok, "Let") ||
         Contains(lastTok, "define") || Contains(lastTok, "Define") ||
         Contains(lastTok, "quantity") || Contains(lastTok, "Quantity") ||
         Contains(lastTok, "symbol") || Contains(lastTok, "Symbol") ||
         Contains(lastTok, "answer") || Contains(lastTok, "Answer") {
        helpers.BoostTokenLogits(lm, ["<<"], 8.0);
      }
      if Contains(lastTok, "final") || Contains(lastTok, "Final") {
        helpers.BoostTokenLogits(lm, [eosToken], 3.0);
      }
    }
    var next := lm.ChooseNextTokenUnconstrained();
    helpers.cost := helpers.cost + 1;
    steps := steps + 1;
    if next == eosToken {
      // stop cleanly on EOS
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
    var isComplete := parser.IsCompletePrefix(currentConstrainedOut);
    if isComplete {
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
      var next := helpers.ConstrainedStep(
        lm, parser, constrainedPrompt, currentConstrainedOut, eosToken
      );
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
    }
  }
}

cost := steps;
  }
}
