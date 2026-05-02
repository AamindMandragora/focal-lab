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
// Strategy for step-by-step math solutions with arithmetic fenced by << >>.
// The state tracks the full generated text in `generated`, whether we are
// currently inside a constrained arithmetic span in `insideConstrainedOut`, and
// the contents of that active span in `currentConstrainedOut`. Outside a span,
// generation is mostly free, but if "<<" is already competitive with the
// current argmax we proactively open a constrained span so arithmetic is more
// likely to be emitted in the required delimiters. Inside a span, we inspect
// parser-valid next-token options: if the active prefix is complete we close
// with ">>"; otherwise we prefer a valid argmax token when available, and fall
// back to constrained decoding when the unconstrained argmax would violate the
// arithmetic parser. This tracked state lets us maintain that the constrained
// suffix of `generated` always matches `currentConstrainedOut`, so parser-based
// validity checks apply exactly to the text between delimiters.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity: In the unconstrained branch, either we append a normal token
//   and remain outside (implication vacuous), or we append "<<" and set
//   currentConstrainedOut := [], which is a valid prefix by precondition. In
//   the complete-prefix branch, CloseConstrainedSpan resets to outside, so the
//   implication is vacuous. In the inside-span argmax-valid branch, we call
//   AppendConstrainedToken only after IsTokenValidNext returned true, so the
//   new constrained prefix remains valid. In the fallback constrained-step
//   branch, if EOS is returned we break without changing state; otherwise
//   ConstrainedStep supplies a parser-valid next token, and AppendConstrainedToken
//   preserves validity.
// suffix: Outside-span branches either leave insideConstrainedOut false, or
//   after appending "<<" set currentConstrainedOut := [], whose length-0 suffix
//   matches generated. CloseConstrainedSpan appends the closing delimiter and
//   resets currentConstrainedOut := [], so the implication holds. Both
//   AppendConstrainedToken branches append the same token to `generated` and to
//   `currentConstrainedOut`, preserving the suffix equality.
// cost: In the unconstrained raw-sampling branch we manually bump
//   helpers.cost by 1 and also increment steps by 1. CloseConstrainedSpan bumps
//   helpers.cost internally and we increment steps once. In the inside-span
//   argmax-valid branch we only use queries plus AppendConstrainedToken, which
//   does not bump helpers.cost, and then increment steps by 1, so
//   helpers.cost <= steps still holds. In the fallback branch, ConstrainedStep
//   bumps helpers.cost by 1 and we increment steps by 1; the subsequent append
//   is non-bumping.
// progress: In the unconstrained branch we append at most one token to
//   generated. In the complete-prefix branch we append exactly the closing
//   delimiter token once. In the inside-span argmax-valid and fallback branches
//   we append at most one constrained token. Since steps increases by exactly 1
//   in every non-breaking iteration, |generated| <= |generatedPrefix| + steps
//   is preserved.
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
    var next := lm.ChooseNextTokenUnconstrained();
    helpers.cost := helpers.cost + 1;
    steps := steps + 1;

    if next == eosToken {
      break;
    } else if openLogit >= argmaxLogit - 2.0 && next != "<<" && !Contains(next, "<<") {
      generated := generated + ["<<"];
      insideConstrainedOut := true;
      currentConstrainedOut := [];
    } else {
      generated := generated + [next];
      if next == "<<" {
        insideConstrainedOut := true;
        currentConstrainedOut := [];
      }
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

      if argmax != eosToken && argmaxValid {
        var appendedGenerated, appendedInside, appendedCurrent := helpers.AppendConstrainedToken(
          lm, parser, generated, currentConstrainedOut, argmax
        );
        generated := appendedGenerated;
        insideConstrainedOut := appendedInside;
        currentConstrainedOut := appendedCurrent;
        steps := steps + 1;
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
