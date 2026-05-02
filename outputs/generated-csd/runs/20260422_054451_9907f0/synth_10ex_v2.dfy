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
// The previous strategy failed because it opened `<<` far too early and then kept
// the model trapped inside constrained mode before any natural-language reasoning
// had been produced. That caused outputs like `Let n<<...` and many unterminated
// spans. The fix is to make constrained decoding sparse, late, and self-closing:
//
// - Outside constrained mode, default behavior is ordinary unconstrained generation.
//   We only open `<<` when the model itself strongly indicates it by making `<<`
//   the highest-logit token. We do not open merely because other tokens contain
//   math-like characters such as `*`, `/`, or `=`.
// - Once inside constrained mode, we keep the span short: if the constrained prefix
//   is complete, we immediately emit `>>`; otherwise we take exactly one
//   parser-valid constrained token. This directly addresses the prior
//   unterminated-span failures.
// - We additionally avoid opening a constrained span near the step budget limit,
//   because opening without enough remaining budget to both emit content and close
//   is risky.
// - EOS remains terminal everywhere.
//
// This revised strategy is conservative: it preserves normal step-by-step text
// generation most of the time, while still allowing verified arithmetic spans when
// the model explicitly chooses to start one.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity: In the outside branch, either UnconstrainedStep keeps us
//   outside constrained mode so the implication is vacuous, or OpenConstrainedSpan
//   starts a fresh constrained segment with empty contents, which is valid because
//   parser.IsValidPrefix([]) holds by precondition. In the inside branch,
//   CloseConstrainedSpan exits constrained mode so the implication becomes
//   vacuous; otherwise ConstrainedStep chooses a parser-valid next token and
//   AppendConstrainedToken extends currentConstrainedOut with that token, so
//   validity is preserved. EOS-break branches do not mutate the constrained state.
// suffix: In any branch that leaves us outside constrained mode, the implication is
//   vacuous. OpenConstrainedSpan appends only the opening delimiter and sets
//   currentConstrainedOut to [], so the generated suffix of length 0 matches.
//   AppendConstrainedToken appends the same token to both generated and the
//   constrained contents, preserving the suffix equality. CloseConstrainedSpan
//   exits constrained mode, making the implication vacuous. Break branches leave
//   generated/currentConstrainedOut unchanged.
// cost: The only helpers here that bump helpers.cost are UnconstrainedStep,
//   OpenConstrainedSpan, ConstrainedStep, and CloseConstrainedSpan. In every
//   branch where one of these is called, steps is incremented exactly once in the
//   same iteration before any further iteration begins. Query/logit-adjustment
//   operations do not bump helpers.cost, and AppendConstrainedToken also does not,
//   so helpers.cost <= steps is preserved.
// progress: Each non-break branch appends at most one token to generated:
//   UnconstrainedStep adds one token, OpenConstrainedSpan adds `<<`,
//   AppendConstrainedToken adds one constrained token, and CloseConstrainedSpan
//   adds `>>`. In each such branch, steps increases by exactly one, so the bound
//   |generated| <= |generatedPrefix| + steps remains true. Break branches stop
//   without further growth.
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
  if insideConstrainedOut {
    var complete := parser.IsCompletePrefix(currentConstrainedOut);
    if complete {
      var closedGenerated, closedInside, closedCurrent := helpers.CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut);
      generated := closedGenerated;
      insideConstrainedOut := closedInside;
      currentConstrainedOut := closedCurrent;
      steps := steps + 1;
    } else {
      var nextIn := helpers.ConstrainedStep(lm, parser, prompt, currentConstrainedOut, eosToken);
      steps := steps + 1;
      if nextIn == eosToken {
        break;
      } else {
        var appendedGenerated, appendedInside, appendedCurrent := helpers.AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, nextIn);
        generated := appendedGenerated;
        insideConstrainedOut := appendedInside;
        currentConstrainedOut := appendedCurrent;
      }
    }
  } else {
    lm.GenerateLogits(prompt + generated);
    helpers.PenalizeTokenLogits(lm, [">>"], 100.0);
    var best := helpers.GetHighestLogitToken(lm);

    var canOpen := false;
    if steps + 2 < maxSteps {
      if best == "<<" {
        canOpen := true;
      }
    }

    if canOpen {
      var openedGenerated, openedInside, openedCurrent := helpers.OpenConstrainedSpan(lm, generated);
      generated := openedGenerated;
      insideConstrainedOut := openedInside;
      currentConstrainedOut := openedCurrent;
      steps := steps + 1;
    } else {
      var next := helpers.UnconstrainedStep(lm, prompt, generated);
      steps := steps + 1;
      if next == eosToken {
        break;
      } else {
        generated := generated + [next];
      }
    }
  }
}

cost := steps;
    if maxSteps > 0 && cost == 0 { cost := 1; }  // guarantee progress postcondition
  }
}
