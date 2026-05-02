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
// Strategy for step-by-step math solutions with arithmetic enclosed in << >>.
// The state tracks the full generated output (`generated`), whether decoding is
// currently inside a constrained arithmetic span (`insideConstrainedOut`), and
// the contents of the active span without delimiters (`currentConstrainedOut`).
// Outside a span, generation is mostly free, but if the model either emits "<<"
// directly or gives it a logit close to the current argmax, we proactively open
// a constrained span so arithmetic is likely to be formatted correctly. Inside
// a span, we keep the parser-valid prefix in `currentConstrainedOut`; if the
// current prefix is complete we close with ">>", otherwise we prefer a valid
// argmax token when available and fall back to `ConstrainedStep` when needed.
// This tracked state ensures that text between delimiters remains parser-valid
// arithmetic while the rest of the explanation can still be generated freely.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity: In the unconstrained eager-open branch we append "<<" to
//   generated and set currentConstrainedOut := [], which is a valid prefix by
//   precondition. In the ordinary unconstrained commit branch we either remain
//   outside the span (making the implication vacuous) or enter on next == "<<"
//   with currentConstrainedOut := []. In the complete-prefix branch,
//   CloseConstrainedSpan sets insideConstrainedOut to false, so the invariant is
//   vacuous afterward. In the constrained argmax-fast-path we first check
//   IsTokenValidNext before AppendConstrainedToken, so the new constrained
//   prefix remains valid. In the constrained fallback branch, ConstrainedStep
//   provides a parser-valid next token unless it returns EOS, and on EOS we
//   break without mutating the tracked constrained prefix.
// suffix: In both span-entry branches, currentConstrainedOut is reset to [],
//   and the length-0 suffix of generated matches []. In the unconstrained
//   outside-span branch, generated changes only by appending one token while
//   currentConstrainedOut stays []; if we enter on "<<", the suffix relation is
//   again immediate with []. In the complete-prefix branch,
//   CloseConstrainedSpan appends ">>" and resets currentConstrainedOut to [],
//   so the implication is vacuous. In both constrained append branches,
//   AppendConstrainedToken appends the same token to generated and to
//   currentConstrainedOut, preserving the suffix equality.
// cost: In the eager-open unconstrained branch we use raw logits and then
//   manually do helpers.cost := helpers.cost + 1 while also incrementing steps,
//   so helpers.cost <= steps is preserved. In the unconstrained sampling branch,
//   UnconstrainedStep bumps helpers.cost by 1 internally and we increment steps
//   once. In the complete-prefix branch, CloseConstrainedSpan bumps cost by 1
//   and we increment steps once. In the constrained argmax-fast-path we only
//   use queries plus AppendConstrainedToken, which does not bump cost, while we
//   still increment steps, so the inequality remains true. In the constrained
//   fallback branch, ConstrainedStep bumps by 1 and we increment steps once.
// progress: In the eager-open branch we append exactly one token ("<<") while
//   steps increases by 1. In the ordinary unconstrained branch we append at
//   most one sampled token while steps increases by 1; if EOS is sampled we
//   break before appending. In the complete-prefix branch we append exactly one
//   closing delimiter while steps increases by 1. In the constrained fast-path
//   and fallback append branches we append at most one token while steps
//   increases by 1; on EOS fallback we break without appending. Therefore
//   |generated| never grows by more than the number of steps taken.
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

    if openLogit >= argmaxLogit - 2.0 && argmax != eosToken && argmax != "<<" {
      helpers.cost := helpers.cost + 1;
      steps := steps + 1;
      generated := generated + ["<<"];
      insideConstrainedOut := true;
      currentConstrainedOut := [];
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
      var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];
      lm.GenerateLogits(constrainedPrompt + currentConstrainedOut);
      var argmax := helpers.GetHighestLogitToken(lm);
      var argmaxValid := false;
      if argmax != eosToken {
        argmaxValid := helpers.IsTokenValidNext(parser, currentConstrainedOut, argmax);
      }

      if argmaxValid {
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
