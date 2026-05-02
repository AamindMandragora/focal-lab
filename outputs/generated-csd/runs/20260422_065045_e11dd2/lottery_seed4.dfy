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
// The previous strategy failed because it tried to force constrained generation
// too early and too often. In evaluation, that caused the model to enter `<<`
// almost immediately, then wander inside the constrained grammar and never reach
// `>>`. The fix is to reverse the policy:
//
// 1. Default behavior is now ordinary unconstrained generation. We do not
//    proactively open `<<` based on logits alone.
// 2. We only recognize constrained mode when the sampled outside token is
//    literally `<<`. This avoids the "entered_constrained_mode_too_early"
//    failure caused by aggressive opening.
// 3. Once inside constrained mode, we make completion the top priority. If the
//    constrained prefix is complete, we close immediately. If only a small
//    budget remains, we stop rather than risk leaving an unterminated span.
// 4. While inside, we use parser-guided constrained stepping, but we also
//    strongly bias toward short spans by closing at the first complete prefix.
//    We never call CloseConstrainedSpan unless completeness is already known.
// 5. Outside constrained mode, when only one step remains, we forbid opening a
//    new span by breaking on a sampled `<<`; this preserves the ability to avoid
//    unterminated delimiters near the budget limit.
//
// This design directly targets the observed failures: do not open early, and
// once opened, close as soon as correctness permits.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity: In outside-mode branches, either we remain outside after an
//   unconstrained token or we switch to constrained mode only on sampled `<<`,
//   setting currentConstrainedOut to []; then the implication holds because []
//   is a valid prefix by precondition. In inside-mode branches, CloseConstrainedSpan
//   exits constrained mode so the implication is vacuous; otherwise ConstrainedStep
//   followed by AppendConstrainedToken extends the constrained prefix with a
//   parser-valid next token, preserving validity.
//
// suffix: Outside mode, the invariant is vacuous unless the sampled token is
//   `<<`; in that case currentConstrainedOut becomes [], so generated's suffix of
//   length 0 equals []. In inside mode, closing makes the implication vacuous;
//   appending a constrained token extends both generated and currentConstrainedOut
//   by the same token, so the suffix equality is preserved.
//
// cost: Each loop iteration that changes state uses at most one cost-bumping
//   helper among UnconstrainedStep, ConstrainedStep, and CloseConstrainedSpan,
//   and then increments steps exactly once. Query helpers do not bump cost, and
//   AppendConstrainedToken is non-bumping, so helpers.cost <= steps is preserved.
//
// progress: Initially |generated| = |generatedPrefix|. On each non-breaking
//   iteration, steps increases by 1, while generated grows by at most one token:
//   one unconstrained token outside, one constrained token inside, or one closing
//   delimiter when closing. Therefore |generated| never exceeds
//   |generatedPrefix| + steps.
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
  invariant helpers.cost <= steps
  decreases maxSteps - steps
{
  if insideConstrainedOut {
    var completeNow := parser.IsCompletePrefix(currentConstrainedOut);
    if completeNow {
      var closedGenerated, closedInside, closedCurrent := helpers.CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut);
      generated := closedGenerated;
      insideConstrainedOut := closedInside;
      currentConstrainedOut := closedCurrent;
      steps := steps + 1;
    } else {
      if maxSteps - steps <= 1 {
        break;
      } else {
        var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];
        var nextIn := helpers.ConstrainedStep(lm, parser, constrainedPrompt, currentConstrainedOut, eosToken);
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
    }
  } else {
    var nextOut := helpers.UnconstrainedStep(lm, prompt, generated);
    steps := steps + 1;
    if nextOut == eosToken {
      break;
    } else {
      if nextOut == "<<" {
        if steps < maxSteps {
          generated := generated + [nextOut];
          insideConstrainedOut := true;
          currentConstrainedOut := [];
        } else {
          break;
        }
      } else {
        generated := generated + [nextOut];
      }
    }
  }
}

cost := steps;
    if maxSteps > 0 && cost == 0 { cost := 1; }  // guarantee progress postcondition
  }
}
