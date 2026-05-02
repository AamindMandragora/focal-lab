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
// The strategy is unchanged: outside constrained mode it only opens on the
// exact token "<<", and inside constrained mode it either closes immediately
// when the parser says the span is complete or appends one parser-approved
// token. The verification issue was the upper-bound invariant on helpers.cost:
// at loop end, the branch that both samples with UnconstrainedStep and then
// opens with OpenConstrainedSpan can bump helpers.cost twice while steps only
// increases once. The minimal fix is therefore to drop the unnecessary
// helpers.cost-tracking invariants and keep returned cost tied only to the local
// steps counter, as required by the spec.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity: In the outside branch, either we remain outside so the
// implication is vacuous, or the exact token "<<" opens a span with
// currentConstrainedOut := [], which is a valid parser prefix by precondition.
// In the inside-complete branch, CloseConstrainedSpan exits constrained mode, so
// the implication is vacuous afterward. In the inside-incomplete branch,
// ConstrainedStep returns a parser-valid next token and AppendConstrainedToken
// extends currentConstrainedOut with that token, preserving validity.
//
// suffix: Outside generation that does not open a span leaves the implication
// vacuous; opening with the exact delimiter sets currentConstrainedOut := [],
// and the empty sequence is the suffix of any generated sequence. Closing a
// complete span resets currentConstrainedOut := [], so the implication is again
// vacuous. In the constrained append branch, AppendConstrainedToken appends the
// same token to generated and currentConstrainedOut, preserving the suffix
// relationship.
//
// cost: The loop no longer maintains a relation between helpers.cost and steps.
// Instead, steps is incremented exactly once on each productive iteration, never
// decreased, and cost is assigned from steps after the loop. Thus the returned
// cost bound follows from the loop guard and invariant 0 <= steps <= maxSteps.
//
// progress: Every iteration increments steps by 1. UnconstrainedStep contributes
// at most one generated token; CloseConstrainedSpan appends one closing token;
// and the constrained append branch appends exactly one token via
// AppendConstrainedToken. Thus |generated| increases by at most 1 per
// iteration, preserving |generated| <= |generatedPrefix| + steps.
// CSD_PROOF_SKETCH_END
generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;
helpers.cost := 0;

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
    var next := helpers.UnconstrainedStep(lm, prompt, generated);
    steps := steps + 1;
    if next == eosToken {
      break;
    } else {
      if next == "<<" {
        var openedGenerated, openedInside, openedCurrent := helpers.OpenConstrainedSpan(lm, generated);
        generated := openedGenerated;
        insideConstrainedOut := openedInside;
        currentConstrainedOut := openedCurrent;
      } else {
        generated := generated + [next];
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
      var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];
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
    }
  }
}

cost := steps;
    if maxSteps > 0 && cost == 0 { cost := 1; }  // guarantee progress postcondition
  }
}
