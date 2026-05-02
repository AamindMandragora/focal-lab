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
    // CSD_RATIONALE_BEGIN
// Targeted failure mode: many failures never entered constrained mode at all,
// producing long unconstrained scratchwork or partial free-form equations instead
// of arithmetic inside << >>. The single edit is to broaden the outside-span
// opening trigger: in addition to opening after an operator immediately before a
// recent ">>", we also open whenever the most recently generated token itself is
// "=". This keeps the previous strategy intact while directly addressing the
// common pattern "name = ..." where the next content should be a constrained
// arithmetic span.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity: In the outside-span branch, staying outside leaves the
//   implication vacuous; opening due to either trigger sets currentConstrainedOut
//   := [], which is parser-valid by the precondition on []. In the complete
//   constrained branch, CloseConstrainedSpan makes insideConstrainedOut false, so
//   the implication is vacuous. In the active constrained append branch,
//   AdaptiveConstrainedStep returns eosToken or a parser-valid next token, and
//   AppendConstrainedToken preserves parser validity of currentConstrainedOut.
// suffix: Outside the span, either state is unchanged or we open a new span with
//   currentConstrainedOut := [], and the empty sequence is the suffix of any
//   generated prefix. CloseConstrainedSpan resets currentConstrainedOut to []
//   after appending at most one delimiter token, so the suffix invariant holds
//   trivially. In the constrained append branch, AppendConstrainedToken appends
//   the same token to generated and currentConstrainedOut, preserving suffix
//   equality.
// cost accounting: We still return cost := steps, and every non-breaking branch
//   increments steps exactly once after a cost-bumping helper call
//   (UnconstrainedStep, OpenConstrainedSpan, CloseConstrainedSpan, or
//   AdaptiveConstrainedStep). Break branches do not mutate further, so the loop
//   bound gives cost <= maxSteps at return.
// progress bound: UnconstrainedStep appends exactly one token and steps is
//   incremented by 1; the open-span branch appends exactly one token and also
//   increments steps by 1. CloseConstrainedSpan appends at most one token and
//   increments steps by 1. AdaptiveConstrainedStep consumes one step; on non-EOS
//   we then append exactly one constrained token with AppendConstrainedToken, and
//   on EOS we break immediately. Thus every non-breaking branch strictly
//   increases steps and preserves |generated| <= |generatedPrefix| + steps.
// CSD_PROOF_SKETCH_END
generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

var steps: nat := 0;
var lastBeforeClose: Token := "";
var sawCloseContext: bool := false;
var narrowThreshold: nat := 12;

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
    var prevTok, foundPrev := helpers.LastTokenBefore(generated, ">>");
    if foundPrev {
      lastBeforeClose := prevTok;
      sawCloseContext := true;
    }

    var shouldOpen := false;
    if sawCloseContext {
      if lastBeforeClose == "=" || lastBeforeClose == "+" || lastBeforeClose == "-" || lastBeforeClose == "*" || lastBeforeClose == "/" {
        shouldOpen := true;
      }
    }
    if !shouldOpen && 0 < |generated| {
      if generated[|generated| - 1] == "=" {
        shouldOpen := true;
      }
    }

    if shouldOpen && "<<" in lm.Tokens {
      var openedGenerated, openedInside, openedCurrent := helpers.OpenConstrainedSpan(lm, generated);
      generated := openedGenerated;
      insideConstrainedOut := openedInside;
      currentConstrainedOut := openedCurrent;
      steps := steps + 1;
      sawCloseContext := false;
    } else {
      var next := helpers.UnconstrainedStep(lm, prompt, generated);
      steps := steps + 1;
      if next == eosToken {
        break;
      } else {
        generated := generated + [next];
        if next == "<<" {
          insideConstrainedOut := true;
          currentConstrainedOut := [];
          sawCloseContext := false;
        }
      }
    }
  } else {
    var complete := parser.IsCompletePrefix(currentConstrainedOut);
    if complete {
      var closedGenerated, closedInside, closedCurrent := helpers.CloseConstrainedSpan(
        lm, parser, generated, currentConstrainedOut
      );
      generated := closedGenerated;
      insideConstrainedOut := closedInside;
      currentConstrainedOut := closedCurrent;
      steps := steps + 1;
    } else {
      var stablePrefix := generated[..|generated| - |currentConstrainedOut|];
      var next := helpers.AdaptiveConstrainedStep(
        lm, parser, prompt + stablePrefix, currentConstrainedOut, validTokenGroups, 4.0, narrowThreshold, eosToken
      );
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
    }
  }
}

cost := steps;
  }
}
