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
// The previous strategy failed because it tried to infer delimiter structure
// from arbitrary unconstrained tokens and even opened a fresh constrained span
// after tokens that merely contained "<<". That produced accidental "<<<<",
// left spans unterminated, and allowed long wandering inside a span.
// 
// This revision makes delimiter handling explicit and conservative:
// 1) Outside a constrained span, generation is unconstrained except that if the
//    sampled token is exactly "<<", we enter constrained mode; otherwise we just
//    append the token and never synthesize or infer an opening delimiter.
// 2) Inside a constrained span, we never sample unconstrained text. If the
//    current constrained prefix is complete, we immediately close with `>>` on
//    the next step. Otherwise we take exactly one parser-valid constrained step
//    and append that token.
// 3) EOS is terminal everywhere: if sampled, we stop immediately rather than
//    leaving partially updated state.
// 
// The key fix for evaluation is that the only way to open a constrained span is
// by actually generating the exact delimiter token "<<", and every complete
// constrained prefix is eagerly closed with `>>`. This directly targets the
// observed unterminated-span and accidental-"<<<<" failures while preserving a
// simple invariant story.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity: In the outside branch, if the token is not "<<", we remain
//   outside so the implication is vacuous; if it is "<<", we enter with
//   `currentConstrainedOut := []`, and `parser.IsValidPrefix([])` holds. In the
//   inside-complete branch, `CloseConstrainedSpan` exits constrained mode, so
//   the implication is vacuous. In the inside-incomplete branch,
//   `ConstrainedStep` returns a parser-valid next token (or EOS, which breaks),
//   and `AppendConstrainedToken` preserves validity of the constrained prefix.
// suffix: Outside with a non-delimiter token, constrained state is unchanged or
//   absent, so the invariant is preserved. When the exact token "<<" is
//   appended, we set `currentConstrainedOut := []`, and the empty suffix matches
//   by definition. Closing a complete span resets constrained state to empty;
//   appending a constrained token extends both `generated` and
//   `currentConstrainedOut` by the same token via `AppendConstrainedToken`.
// cost: Each loop iteration performs at most one cost-bumping helper:
//   `UnconstrainedStep`, `ConstrainedStep`, or `CloseConstrainedSpan`. We then
//   increment `steps` exactly once in that iteration. `AppendConstrainedToken`
//   is non-bumping, so after every branch `helpers.cost <= steps` still holds.
// progress: In the outside branch we append at most one token after one step.
//   In the inside-complete branch, `CloseConstrainedSpan` appends only `>>` and
//   we increment `steps` once. In the inside-incomplete branch, one constrained
//   token is appended after one constrained step. Thus every iteration grows
//   `generated` by at most one token while `steps` grows by one, preserving
//   `|generated| <= |generatedPrefix| + steps`.
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
      var next := helpers.ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken);
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
