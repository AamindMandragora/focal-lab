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
// This strategy tracks the full generated output in `generated`, whether we are
// currently inside a math-computation span in `insideConstrainedOut`, and the
// current contents of that span (excluding delimiters) in
// `currentConstrainedOut`. Outside constrained spans it generates mostly
// freely, but it proactively opens a span when either the sampled token is "<<"
// or the open-delimiter logit is close to the current argmax, which encourages
// step-by-step arithmetic to be written inside delimiters.
// Inside a constrained span, the strategy uses parser-aware generation. If the
// current constrained prefix is already complete, it closes with ">>". If not,
// it first checks whether the unconstrained argmax is parser-valid and uses it
// as a fast path; otherwise it falls back to `ConstrainedStep`. This tracked
// state ensures that only parser-valid tokens extend the constrained segment,
// and that the generated suffix always matches the constrained contents.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity: In the unconstrained branch, committing a normal token keeps
//   `insideConstrainedOut == false`, so the implication is vacuous; opening a
//   span sets `currentConstrainedOut := []`, which is valid by precondition.
//   In the complete-prefix branch, `CloseConstrainedSpan` resets
//   `insideConstrainedOut` to false, again making the implication vacuous. In
//   the constrained fast-path we call `AppendConstrainedToken` only after
//   `IsTokenValidNext` says the argmax is valid; in the fallback path,
//   `ConstrainedStep` supplies a valid next token (or EOS, which breaks), so
//   appending preserves parser validity.
// suffix: Unconstrained normal-token commits leave the antecedent false; when
//   we open a span we append "<<" to `generated` and set
//   `currentConstrainedOut := []`, so the length-0 suffix matches. Closing via
//   `CloseConstrainedSpan` appends ">>" and resets the constrained prefix to
//   [], so the implication holds. In both constrained append branches,
//   `AppendConstrainedToken` appends the same token to `generated` and to
//   `currentConstrainedOut`, preserving the suffix equality.
// cost: In the unconstrained raw-sampling branch we use `GenerateLogits` plus
//   `ChooseNextTokenUnconstrained`, then manually do `helpers.cost :=
//   helpers.cost + 1` and `steps := steps + 1`; thus helpers-cost grows by at
//   most one per iteration. `CloseConstrainedSpan` and `ConstrainedStep` each
//   bump helpers-cost by 1 internally, and we also increment `steps` once in
//   those branches. The constrained fast-path only increments `steps`; its
//   queries and `AppendConstrainedToken` do not bump helpers-cost, so
//   `helpers.cost <= steps` is preserved.
// progress: In every non-break branch, `steps` increases by exactly 1. The
//   unconstrained branch appends at most one token; opening a span appends only
//   the single token "<<". The close branch appends only ">>". Each constrained
//   append branch appends exactly one token, so `|generated|` never increases
//   by more than `steps` beyond `generatedPrefix`.
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
    } else if next == "<<" {
      generated := generated + [next];
      insideConstrainedOut := true;
      currentConstrainedOut := [];
    } else if openLogit >= argmaxLogit - 2.0 && !Contains(next, "<<") {
      generated := generated + ["<<"];
      insideConstrainedOut := true;
      currentConstrainedOut := [];
    } else {
      generated := generated + [next];
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
