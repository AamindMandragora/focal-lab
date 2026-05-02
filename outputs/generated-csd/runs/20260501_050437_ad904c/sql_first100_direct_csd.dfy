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
    stepTokenBudget: nat,
    validTokenGroups: seq<seq<Token>>,
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
// Revised Spider text-to-SQL strategy: force a single constrained SQL span as
// early as possible, keep decoding almost entirely parser-driven, and add only
// lightweight SQL-specific steering that discourages the repetition loops seen
// in the previous attempt.
//
// What changed from the failed version:
// 1. Outside the constrained span, we no longer wander unconstrained for many
//    tokens. If already outside, we immediately open the SQL span with "<<"
//    using OpenConstrainedSpan. This guarantees the model spends its budget on
//    SQL tokens rather than free-form text.
// 2. Inside the span, the default action is a single constrained step via the
//    verified helpers, not repeated manual per-step boosting of many groups.
//    This removes the huge BoostTokenLogits overhead and the runaway bias
//    toward copying schema identifiers.
// 3. We use a small anti-repetition heuristic: when the last generated
//    constrained token is known, we penalize that exact token on the next step.
//    The verification fix here is minimal: we now guard that penalized branch
//    with `lastTok in lm.Tokens`, and otherwise fall back to the plain
//    ConstrainedStep branch. This is enough to prove the helper precondition
//    that every penalized token belongs to `lm.Tokens`.
// 4. We still exploit caller-supplied validTokenGroups, but only when the
//    grammar is relatively narrow, via GroupBoostedConstrainedStep. In wider
//    states we avoid expensive broad boosting and instead use either a plain
//    ConstrainedStep or a single-token penalty.
// 5. We close the constrained span immediately once the parser says the SQL
//    prefix is complete. EOS inside the span remains terminal and causes an
//    immediate stop.
//
// Overall, the strategy is now: open span immediately, decode SQL under hard
// parser control, lightly boost schema groups only in narrow states, penalize
// immediate repetition when the token is known to be in the LM vocabulary, and
// close as soon as a complete SQL query is formed.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity: Opening the span sets currentConstrainedOut := [], which is
//   parser-valid by precondition. The close-span branch makes
//   insideConstrainedOut false, so the implication is vacuous. In the plain
//   constrained-step branch, ConstrainedStep preserves parser validity on every
//   non-EOS step. In the group-boosted and penalized branches, the helper
//   returns either eosToken or a parser-valid next token; on non-EOS,
//   AppendConstrainedToken preserves parser validity. In the new fallback
//   branch for `lastTok !in lm.Tokens`, we use ConstrainedStep, which also
//   preserves parser validity.
// suffix: Opening the span leaves currentConstrainedOut empty, so the suffix
//   property holds trivially. Closing the span resets insideConstrainedOut to
//   false, making the implication vacuous. ConstrainedStep preserves the suffix
//   invariant by specification. In the group-boosted and penalized branches,
//   AppendConstrainedToken appends the same chosen token to both generated and
//   currentConstrainedOut, so the active constrained suffix remains aligned.
//   The fallback plain-step branch preserves the suffix by the ConstrainedStep
//   postcondition.
// cost accounting: cost is not used during the loop and is assigned from steps
//   at the end. Every non-breaking branch calls exactly one cost-bumping helper
//   (OpenConstrainedSpan, CloseConstrainedSpan, ConstrainedStep,
//   GroupBoostedConstrainedStep, or PenalizedConstrainedStep) and then does
//   steps := steps + 1. Breaking branches also occur only after such a helper,
//   so steps never exceeds maxSteps and returned cost := steps is bounded.
// progress: OpenConstrainedSpan appends exactly one token and we increment
//   steps by 1. CloseConstrainedSpan appends at most one token and again steps
//   grows by 1. ConstrainedStep either breaks on EOS after one consumed step or
//   appends exactly one token and steps increases by 1. GroupBoosted and
//   penalized branches each consume one constrained sampling step; on non-EOS
//   they append exactly one token via AppendConstrainedToken, and on EOS they
//   break immediately. The new fallback branch is also a one-step
//   ConstrainedStep branch, so every non-breaking branch strictly increases
//   steps, preserving |generated| <= |generatedPrefix| + steps.
// CSD_PROOF_SKETCH_END
generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

var steps: nat := 0;
var narrowThreshold: nat := 10;

while steps < maxSteps
  invariant 0 <= steps <= maxSteps
  invariant lm.ValidTokensIdsLogits()
  invariant !insideConstrainedOut ==> currentConstrainedOut == []
  invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
  invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|
  invariant insideConstrainedOut ==> generated[|generated| - |currentConstrainedOut|..] == currentConstrainedOut
  invariant |generated| <= |generatedPrefix| + steps
  invariant cost == 0
  decreases maxSteps - steps
{
  if !insideConstrainedOut {
    var openedGenerated, openedInside, openedCurrent := helpers.OpenConstrainedSpan(lm, generated);
    generated := openedGenerated;
    insideConstrainedOut := openedInside;
    currentConstrainedOut := openedCurrent;
    steps := steps + 1;
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
      var stablePrefix := generated[..|generated| - |currentConstrainedOut|];
      var constrainedPrompt := prompt + stablePrefix;
      var validCount := helpers.ValidTokenCount(parser, currentConstrainedOut);

      if |currentConstrainedOut| > 0 {
        var lastTok := currentConstrainedOut[|currentConstrainedOut| - 1];
        if validCount <= narrowThreshold {
          var next1 := helpers.GroupBoostedConstrainedStep(
            lm, parser, constrainedPrompt, currentConstrainedOut, validTokenGroups, 3.0, eosToken
          );
          steps := steps + 1;
          if next1 == eosToken {
            break;
          } else {
            var appendedGenerated1, appendedInside1, appendedCurrent1 := helpers.AppendConstrainedToken(
              lm, parser, generated, currentConstrainedOut, next1
            );
            generated := appendedGenerated1;
            insideConstrainedOut := appendedInside1;
            currentConstrainedOut := appendedCurrent1;
          }
        } else {
          if lastTok in lm.Tokens {
            var penalizeTokens := [lastTok];
            var next2 := helpers.PenalizedConstrainedStep(
              lm, parser, constrainedPrompt, currentConstrainedOut, penalizeTokens, 8.0, eosToken
            );
            steps := steps + 1;
            if next2 == eosToken {
              break;
            } else {
              var appendedGenerated2, appendedInside2, appendedCurrent2 := helpers.AppendConstrainedToken(
                lm, parser, generated, currentConstrainedOut, next2
              );
              generated := appendedGenerated2;
              insideConstrainedOut := appendedInside2;
              currentConstrainedOut := appendedCurrent2;
            }
          } else {
            var steppedGenerated2, steppedInside2, steppedCurrent2, hitEos2 := helpers.ConstrainedStep(
              lm, parser, constrainedPrompt, generated, currentConstrainedOut, eosToken
            );
            steps := steps + 1;
            if hitEos2 {
              break;
            } else {
              generated := steppedGenerated2;
              insideConstrainedOut := steppedInside2;
              currentConstrainedOut := steppedCurrent2;
            }
          }
        }
      } else {
        if validCount <= narrowThreshold {
          var next3 := helpers.GroupBoostedConstrainedStep(
            lm, parser, constrainedPrompt, currentConstrainedOut, validTokenGroups, 3.0, eosToken
          );
          steps := steps + 1;
          if next3 == eosToken {
            break;
          } else {
            var appendedGenerated3, appendedInside3, appendedCurrent3 := helpers.AppendConstrainedToken(
              lm, parser, generated, currentConstrainedOut, next3
            );
            generated := appendedGenerated3;
            insideConstrainedOut := appendedInside3;
            currentConstrainedOut := appendedCurrent3;
          }
        } else {
          var steppedGenerated, steppedInside, steppedCurrent, hitEos := helpers.ConstrainedStep(
            lm, parser, constrainedPrompt, generated, currentConstrainedOut, eosToken
          );
          steps := steps + 1;
          if hitEos {
            break;
          } else {
            generated := steppedGenerated;
            insideConstrainedOut := steppedInside;
            currentConstrainedOut := steppedCurrent;
          }
        }
      }
    }
  }
}

cost := steps;
  }
}
