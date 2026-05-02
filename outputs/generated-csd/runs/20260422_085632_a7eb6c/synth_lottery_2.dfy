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
// The state tracks the full generated answer in `generated`, whether we are
// currently inside a constrained arithmetic span in `insideConstrainedOut`, and
// the contents of that active span in `currentConstrainedOut`. Outside a span,
// generation is mostly free, but if the model appears ready to start a
// computation (either by emitting "<<" directly or by giving it a logit close
// to the current argmax), we proactively open a constrained span. Inside a
// span, we prefer a cheap speculative argmax append when that token is parser-
// valid; otherwise we fall back to `ConstrainedStep`, and once the parser says
// the arithmetic fragment is complete we close with ">>". This tracked state
// lets the parser validate only the arithmetic interior while `generated` keeps
// the full answer text including delimiters.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity: In the unconstrained branch, if we proactively open or
//   observe next == "<<", we set currentConstrainedOut := [], which is valid by
//   precondition parser.IsValidPrefix([]); otherwise insideConstrainedOut stays
//   false, so the implication is vacuous. In the complete-prefix branch,
//   CloseConstrainedSpan returns outside the span, again making the implication
//   vacuous. In the speculative constrained branch, we call
//   AppendConstrainedToken only after IsTokenValidNext returned true, so the
//   new currentConstrainedOut remains a valid prefix. In the fallback branch,
//   ConstrainedStep supplies a parser-valid next token (or eosToken, which
//   breaks before mutation), and AppendConstrainedToken preserves validity.
// suffix: In unconstrained generation, appending a normal token leaves us
//   outside the span; if we open a span, currentConstrainedOut is [] so the
//   length-0 suffix condition holds immediately after appending "<<". In the
//   complete-prefix branch, CloseConstrainedSpan appends ">>" and resets the
//   constrained prefix to [], so the suffix condition holds trivially. In both
//   constrained append branches, AppendConstrainedToken appends the same token
//   to generated and currentConstrainedOut, preserving the suffix equality.
// cost: In the unconstrained raw-sampling branch we manually increment
//   helpers.cost once after ChooseNextTokenUnconstrained and also increment
//   steps once, so helpers.cost <= steps is preserved. CloseConstrainedSpan and
//   ConstrainedStep each bump helpers.cost by 1 internally, and we pair each
//   call with exactly one steps increment. The speculative append branch uses
//   only queries plus AppendConstrainedToken, none of which bump helpers.cost,
//   while steps still increases by 1, so the inequality remains true.
// progress: Every loop iteration increments steps by 1. The unconstrained
//   branch appends at most one token to generated; the complete-prefix branch
//   appends only the closing delimiter; the speculative and fallback
//   constrained branches append at most one constrained token. Hence generated
//   grows by at most one token per step, maintaining
//   |generated| <= |generatedPrefix| + steps.
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
    } else if openLogit >= argmaxLogit - 2.0 && next != "<<" {
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

      if argmaxValid && argmax != eosToken {
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
