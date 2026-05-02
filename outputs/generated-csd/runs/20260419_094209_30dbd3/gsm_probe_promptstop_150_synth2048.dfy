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
// This strategy uses a simple verifier-friendly split between unconstrained and
// constrained decoding. Outside << >> spans, it decodes one token at a time and
// immediately stops on EOS. When it sees "<<", it enters constrained mode and
// resets the tracked constrained contents to the empty sequence.
// Inside constrained mode, it only uses parser-guided constrained steps and
// helper transitions that preserve the invariant that currentConstrainedOut is
// exactly the suffix of generated inside the active span. If the constrained
// prefix is already complete, it closes the span. If the parser reports a dead
// end, it rolls back to a valid constrained prefix. Otherwise it takes a
// constrained step, appends a valid token when possible, and rolls back on any
// unexpected invalid choice.
// This avoids helpers with stronger token-membership preconditions on the full
// generated sequence, while still maintaining valid << >> spans and clean EOS
// termination.
// CSD_RATIONALE_END
generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

var steps := 0;

while steps < maxSteps
  invariant 0 <= steps <= maxSteps
  invariant lm.ValidTokensIdsLogits()
  invariant |generated| <= |generatedPrefix| + steps
  invariant !insideConstrainedOut ==> currentConstrainedOut == []
  invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
  invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|
  invariant insideConstrainedOut ==> generated[|generated| - |currentConstrainedOut|..] == currentConstrainedOut
  invariant cost == 0
  decreases maxSteps - steps
{
  if !insideConstrainedOut {
    var next := helpers.UnconstrainedStep(lm, prompt, generated);
    steps := steps + 1;
    if next == eosToken {
    } else {
      generated := generated + [next];
      if next == "<<" {
        insideConstrainedOut := true;
        currentConstrainedOut := [];
      } else {
        if Contains(next, ">>") {
          insideConstrainedOut := false;
          currentConstrainedOut := [];
        }
      }
    }
    if next == eosToken {
      steps := maxSteps;
    }
  } else {
    var isComplete := parser.IsCompletePrefix(currentConstrainedOut);
    if isComplete {
      var closedGenerated, closedInside, closedCurrent := helpers.CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut);
      generated := closedGenerated;
      insideConstrainedOut := closedInside;
      currentConstrainedOut := closedCurrent;
      steps := steps + 1;
    } else {
      var deadEnd := helpers.DeadEndDetection(parser, currentConstrainedOut, 1);
      if deadEnd {
        var stablePrefix := generated[..|generated| - |currentConstrainedOut|];
        var repairedGenerated, repairedCurrent := helpers.RollbackConstrainedSpan(parser, stablePrefix, generated, currentConstrainedOut);
        generated := repairedGenerated;
        currentConstrainedOut := repairedCurrent;
        insideConstrainedOut := true;
        steps := steps + 1;
      } else {
        var next := helpers.ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken);
        steps := steps + 1;
        if next == eosToken {
          steps := maxSteps;
        } else {
          var valid := helpers.IsTokenValidNext(parser, currentConstrainedOut, next);
          if valid {
            var appendedGenerated, appendedInside, appendedCurrent := helpers.AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, next);
            generated := appendedGenerated;
            insideConstrainedOut := appendedInside;
            currentConstrainedOut := appendedCurrent;
          } else {
            var stablePrefix2 := generated[..|generated| - |currentConstrainedOut|];
            var repairedGenerated2, repairedCurrent2 := helpers.RollbackConstrainedSpan(parser, stablePrefix2, generated, currentConstrainedOut);
            generated := repairedGenerated2;
            currentConstrainedOut := repairedCurrent2;
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
