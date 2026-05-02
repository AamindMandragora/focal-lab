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
// Step-by-step math CSD with eager delimiter entry and validity-aware constrained
// decoding. The strategy tracks the full generated output in `generated`, a
// boolean `insideConstrainedOut` indicating whether we are currently between
// `<<` and `>>`, and `currentConstrainedOut`, which stores only the active
// arithmetic substring inside the delimiters.
// 
// Outside constrained mode, the strategy looks at the model logits and opens a
// constrained span eagerly when either the argmax is "<<" or the logit for "<<"
// is close to the argmax. This biases the model toward actually writing
// arithmetic computations inside delimiters. Inside constrained mode, if the
// current constrained prefix is complete, we close it immediately; otherwise we
// try a speculative argmax token if it is parser-valid, and fall back to
// `ConstrainedStep` when it is not.
// 
// The tracked state supports valid generation because delimiter tokens are kept
// only in `generated`, while `currentConstrainedOut` stores just the parser-
// checked interior content. This separation makes it straightforward to prove
// parser validity of the active span and the suffix relationship between the
// active constrained content and the full generated output.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity: In the unconstrained eager-entry branch we append "<<" to
//   `generated`, set `insideConstrainedOut := true`, and reset
//   `currentConstrainedOut := []`; the method precondition gives
//   parser.IsValidPrefix([]). In the ordinary unconstrained commit branch we
//   either remain outside constrained mode (implication vacuous) or enter on
//   next == "<<" with currentConstrainedOut reset to []. In the complete-prefix
//   branch, CloseConstrainedSpan returns outside constrained mode, so the
//   implication is vacuous. In the speculative constrained branch we call
//   AppendConstrainedToken only after IsTokenValidNext returned true, and in the
//   fallback branch ConstrainedStep supplies a parser-valid next token (unless
//   it returns EOS, in which case we break without changing the state).
// suffix: Every unconstrained branch that enters constrained mode does so by
//   appending only the delimiter "<<" to `generated` and setting the active
//   constrained content to [], whose length-0 suffix matches trivially. If we
//   stay outside constrained mode, the implication is vacuous. CloseConstrainedSpan
//   appends ">>" and resets currentConstrainedOut to [], so the implication is
//   again vacuous. In both constrained append branches, AppendConstrainedToken
//   appends the same token to `generated` and `currentConstrainedOut`, preserving
//   the suffix equality.
// cost: In the eager-unconstrained raw-logit branch we use GenerateLogits and
//   logit queries, then manually bump helpers.cost by 1 exactly when we consume
//   one decoding step and also increment `steps`. CloseConstrainedSpan bumps
//   helpers.cost internally and we increment `steps` once in that branch.
//   Speculative constrained append uses only non-bumping queries plus
//   AppendConstrainedToken, then increments `steps`, so helpers.cost <= steps
//   remains true with slack. The constrained fallback uses ConstrainedStep,
//   which bumps internally, and we increment `steps` once as well.
// progress: In the eager-entry branch we append at most one token ("<<"). In
//   the ordinary unconstrained branch we append at most one sampled token. In
//   the close branch we append only ">>". In speculative and fallback
//   constrained branches, AppendConstrainedToken appends one token, while the
//   EOS-break path appends none. Since `steps` increases by exactly 1 on every
//   non-breaking iteration, |generated| stays within |generatedPrefix| + steps.
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
    } else if argmax == "<<" || openLogit >= argmaxLogit - 2.0 {
      generated := generated + ["<<"];
      insideConstrainedOut := true;
      currentConstrainedOut := [];
    } else {
      generated := generated + [argmax];
      if argmax == "<<" {
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
