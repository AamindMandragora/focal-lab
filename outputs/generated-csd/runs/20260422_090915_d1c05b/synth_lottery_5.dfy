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
// The previous strategy opened constrained spans too eagerly and then tried to
// let the parser drive arbitrary math text, which led to many unterminated
// `<<` segments and poor syntax. This revision keeps that overall strategy but
// fixes the specific typing mistake in the outside branch: UnconstrainedStep
// returns the next token, not an updated Prefix. So we now append that token to
// generated explicitly before testing EOS or deciding to reinterpret it as an
// opening delimiter.
// Outside a span, the strategy still only opens when the model strongly
// indicates the exact delimiter token "<<" (either as argmax, as the sampled
// token, or with a logit very close to the argmax). Otherwise it takes a normal
// unconstrained step. Inside a span, if the constrained prefix is complete, it
// closes immediately; otherwise it asks ConstrainedStep for a parser-valid next
// token, appends it, and repeats. EOS remains terminal.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity: In the outside branch, after an unconstrained step we remain
// outside unless we open a span; if we open, currentConstrainedOut is set to [],
// which is a valid parser prefix by precondition. In the inside-complete branch,
// CloseConstrainedSpan leaves constrained mode, so the implication is vacuous.
// In the inside-incomplete branch, eosToken causes a break with no constrained
// mutation; otherwise ConstrainedStep provides a parser-valid token and
// AppendConstrainedToken preserves validity.
//
// suffix: In the outside branch, the invariant is vacuous unless we open a
// span; opening appends only the delimiter token and sets currentConstrainedOut
// to [], so the empty suffix matches. In the inside-complete branch,
// CloseConstrainedSpan exits constrained mode, making the implication vacuous.
// In the inside-incomplete branch, eosToken leaves state unchanged; otherwise
// AppendConstrainedToken appends the same token to generated and
// currentConstrainedOut, preserving the suffix equality.
//
// cost: In the outside branch, UnconstrainedStep bumps helpers.cost by 1 and we
// increment steps by 1. In the inside-complete branch, CloseConstrainedSpan
// bumps helpers.cost by 1 and we increment steps by 1. In the
// inside-incomplete branch, ConstrainedStep bumps helpers.cost by 1 and we
// increment steps by 1; AppendConstrainedToken is non-bumping, so
// helpers.cost <= steps is preserved.
//
// progress: In the outside branch, we append exactly one sampled token to
// generated and then increment steps by 1; if we reinterpret that token as "<<",
// generated still grows by only one token. In the inside-complete branch,
// CloseConstrainedSpan appends one closing delimiter and steps increases by 1.
// In the inside-incomplete branch, eosToken breaks without changing generated;
// otherwise AppendConstrainedToken adds exactly one token after the single step
// increment, preserving |generated| <= |generatedPrefix| + steps.
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
    lm.GenerateLogits(prompt + generated);
    var argmax := helpers.GetHighestLogitToken(lm);
    var argmaxLogit := helpers.GetTokenLogit(lm, argmax);
    var openLogit := helpers.GetTokenLogit(lm, "<<");

    var next := helpers.UnconstrainedStep(lm, prompt, generated);
    generated := generated + [next];
    steps := steps + 1;

    if next == eosToken {
      break;
    } else if next == "<<" || argmax == "<<" || openLogit >= argmaxLogit - 0.5 {
      generated := generated[..|generated| - 1] + ["<<"];
      insideConstrainedOut := true;
      currentConstrainedOut := [];
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
