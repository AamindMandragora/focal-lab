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
// Math-step CSD with eager delimiter entry and parser-guided constrained spans.
// The strategy tracks the full generated text in `generated`, whether we are
// currently inside a computation span in `insideConstrainedOut`, and the active
// span contents alone in `currentConstrainedOut`. Outside a span, it generates
// mostly freely but proactively opens `<<` when the model either proposes it
// directly or gives it a logit close to the current argmax; this encourages the
// model to place arithmetic work inside delimiters. Inside a span, it uses the
// parser state to decide whether the span is complete and can be closed, and
// otherwise prefers parser-valid next tokens by checking the argmax first and
// falling back to a constrained helper when needed. This tracked state makes it
// possible to maintain that the active constrained suffix is always a valid
// parser prefix and exactly matches the suffix of the emitted text between the
// delimiters.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity: In the unconstrained branch, if we append a normal token we
//   keep insideConstrainedOut false, so the implication is vacuous; if we open
//   a span (either because argmax is "<<" or because "<<" is competitive) we
//   set currentConstrainedOut := [], which is valid by the method precondition
//   on parser.IsValidPrefix([]). In the complete-prefix branch,
//   CloseConstrainedSpan resets insideConstrainedOut to false, so validity is
//   again vacuous. In the speculative constrained branch, we call
//   AppendConstrainedToken only after IsTokenValidNext says the argmax is valid;
//   in the fallback branch, ConstrainedStep supplies a valid next token (or EOS,
//   which breaks), so the appended constrained prefix remains valid.
// suffix: In unconstrained normal-token steps we do not change
//   currentConstrainedOut and remain outside a span; when we open a span we
//   append "<<" to generated and set currentConstrainedOut := [], so the
//   length-0 suffix equality holds. CloseConstrainedSpan appends ">>" and
//   clears currentConstrainedOut atomically. In both constrained append
//   branches, AppendConstrainedToken appends the same token to generated and to
//   currentConstrainedOut, preserving the suffix match.
// cost: In the unconstrained raw-logit branch we use GenerateLogits and query
//   helpers without cost, then manually increment helpers.cost by 1 exactly when
//   we commit one sampled/selected step and also increment steps by 1. In the
//   complete-prefix branch, CloseConstrainedSpan bumps helpers.cost by 1 and we
//   also do steps := steps + 1. In the speculative constrained argmax-valid
//   branch we manually increment helpers.cost by 1 before appending and also
//   increment steps by 1; AppendConstrainedToken itself does not bump. In the
//   fallback branch, ConstrainedStep bumps helpers.cost by 1 and we increment
//   steps by 1, so helpers.cost <= steps is preserved in every branch.
// progress: In the unconstrained branch we append at most one token to
//   generated; in the open-span subcase that one token is "<<". In the
//   complete-prefix branch we append exactly one token, namely ">>". In each
//   constrained append branch we append exactly one constrained token, while the
//   EOS branches break without changing generated. Since steps increases by 1 on
//   every non-breaking iteration, |generated| <= |generatedPrefix| + steps is
//   maintained.
// CSD_PROOF_SKETCH_END
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
    var argmax := helpers.GetHighestLogitToken(lm);
    var argmaxLogit := helpers.GetTokenLogit(lm, argmax);
    var openLogit := helpers.GetTokenLogit(lm, "<<");

    helpers.cost := helpers.cost + 1;
    steps := steps + 1;

    if argmax == eosToken {
      break;
    } else if argmax == "<<" {
      generated := generated + ["<<"];
      insideConstrainedOut := true;
      currentConstrainedOut := [];
    } else if openLogit >= argmaxLogit - 2.0 {
      generated := generated + ["<<"];
      insideConstrainedOut := true;
      currentConstrainedOut := [];
    } else {
      generated := generated + [argmax];
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
      var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];
      lm.GenerateLogits(constrainedPrompt + currentConstrainedOut);
      var argmax := helpers.GetHighestLogitToken(lm);
      var argmaxValid := helpers.IsTokenValidNext(parser, currentConstrainedOut, argmax);

      if argmaxValid && argmax != eosToken {
        helpers.cost := helpers.cost + 1;
        steps := steps + 1;
        var appendedGenerated, appendedInside, appendedCurrent := helpers.AppendConstrainedToken(
          lm, parser, generated, currentConstrainedOut, argmax
        );
        generated := appendedGenerated;
        insideConstrainedOut := appendedInside;
        currentConstrainedOut := appendedCurrent;
      } else {
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
}

cost := steps;
    if maxSteps > 0 && cost == 0 { cost := 1; }  // guarantee progress postcondition
  }
}
