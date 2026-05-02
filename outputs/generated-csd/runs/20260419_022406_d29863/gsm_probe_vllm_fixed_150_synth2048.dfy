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
// This strategy tracks the full generated answer, whether generation is currently
// inside a << >> constrained span, and the current constrained-span contents alone.
// Outside constrained spans it generates freely so the model can produce natural
// GSM-style reasoning text and decide when to open a symbolic span with <<.
// Inside constrained spans it uses parser-guided constrained decoding, preferring
// valid next symbolic tokens and closing the span as soon as the parser reports a
// complete symbolic quantity expression. It also checks for dead ends and rolls
// back to a valid constrained prefix when necessary.
// This state separation is important because generated stores the whole answer
// including delimiters, while currentConstrainedOut stores only the symbolic
// content between delimiters. That makes parser validity obligations easy to
// maintain and ensures constrained spans remain valid while EOS always stops
// generation immediately.
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
    var next := helpers.UnconstrainedStep(lm, prompt, generated);
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
      var narrow := helpers.DeadEndDetection(parser, currentConstrainedOut, 1);
      if narrow {
        var stablePrefix := generated[..|generated| - |currentConstrainedOut|];
        var repairedGenerated, repairedCurrent := helpers.RollbackConstrainedSpan(
          parser, stablePrefix, generated, currentConstrainedOut
        );
        generated := repairedGenerated;
        currentConstrainedOut := repairedCurrent;
        steps := steps + 1;
        if parser.IsCompletePrefix(currentConstrainedOut) {
          var closedGenerated2, closedInside2, closedCurrent2 := helpers.CloseConstrainedSpan(
            lm, parser, generated, currentConstrainedOut
          );
          generated := closedGenerated2;
          insideConstrainedOut := closedInside2;
          currentConstrainedOut := closedCurrent2;
        } else {
          insideConstrainedOut := true;
        }
      } else {
        var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];
        var next, isValid := helpers.SoftConstrainedStep(
          lm, parser, constrainedPrompt, currentConstrainedOut, 3.0, eosToken
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
          } else {
            var stablePrefix2 := generated[..|generated| - |currentConstrainedOut|];
            var repairedGenerated2, repairedCurrent2 := helpers.RollbackConstrainedSpan(
              parser, stablePrefix2, generated, currentConstrainedOut
            );
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
