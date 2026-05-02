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
// This strategy tracks the full generated prefix, whether decoding is currently
// inside a constrained << >> span, and the current constrained substring itself.
// Outside constrained spans it performs ordinary decoding, but it biases toward
// EOS after likely answer-completion cues such as "=" or tokens mentioning
// "answer"/"final", so GSM-style outputs stop cleanly once the answer is given.
// Inside constrained spans it preserves parser validity by only appending valid
// symbolic tokens, prefers parser-valid continuations, and closes the span as
// soon as the constrained prefix is complete. If the constrained region appears
// narrow, it uses a stricter constrained step; otherwise it uses a soft
// constrained step and appends only when the chosen token is parser-valid.
// This state supports valid symbolic quantity-name generation because the active
// constrained substring is always maintained as a parser-valid prefix and as the
// suffix of the full generated output while the span is open.
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
  invariant helpers.cost <= steps
  decreases maxSteps - steps
{
  if !insideConstrainedOut {
    lm.GenerateLogits(prompt + generated);
    if |generated| > 0 {
      var lastTok := generated[|generated| - 1];
      if Contains(lastTok, "answer") || Contains(lastTok, "Answer") ||
         Contains(lastTok, "final") || Contains(lastTok, "Final") ||
         Contains(lastTok, "=") {
        helpers.BoostTokenLogits(lm, [eosToken], 3.0);
      }
    }
    var next := lm.ChooseNextTokenUnconstrained();
    helpers.cost := helpers.cost + 1;
    steps := steps + 1;
    if next == eosToken {
      break;
    } else {
      generated := generated + [next];
      if next == "<<" {
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
      var stablePrefix := generated[..|generated| - |currentConstrainedOut|];
      var constrainedPrompt := prompt + stablePrefix;
      var narrow := helpers.DeadEndDetection(parser, currentConstrainedOut, 2);
      if narrow {
        var next := helpers.ConstrainedStep(lm, parser, constrainedPrompt, currentConstrainedOut, eosToken);
        steps := steps + 1;
        if next == eosToken {
          break;
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
          lm, parser, constrainedPrompt, currentConstrainedOut, 2.0, eosToken
        );
        steps := steps + 1;
        if next == eosToken {
          break;
        } else {
          if isValid {
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
  }
}

cost := steps;
  }
}
