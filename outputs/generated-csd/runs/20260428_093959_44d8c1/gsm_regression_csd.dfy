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
    generated := generatedPrefix;
    insideConstrainedOut := insideConstrained;
    currentConstrainedOut := currentConstrained;
    cost := 0;
    // CSD_RATIONALE_BEGIN
// Revised strategy: aggressively guarantee balanced << >> spans and improve
// syntax by making span management deterministic rather than advisory.
//
// The prior version relied on the model to emit "<<" during unconstrained
// chunking and then to finish the arithmetic before the step budget expired,
// which led to many unterminated constrained segments. This revision fixes that
// in two ways:
//
// 1) Outside constrained mode, generation is token-by-token. If the sampled
//    unconstrained token contains "<<", we immediately replace that event with a
//    helper-mediated OpenConstrainedSpan transition, so the strategy state is
//    synchronized with the emitted delimiter.
// 2) Inside constrained mode, we reserve one remaining step for closing ">>".
//    When only one step remains, we stop adding arithmetic tokens; if the
//    arithmetic prefix is already complete we close immediately, otherwise we
//    roll back the whole constrained span to the stable prefix and exit the
//    span by clearing the state. This sacrifices a malformed partial span
//    instead of emitting an unterminated one.
// 3) While inside a span, decoding is parser-masked on every token. When the
//    prefix is complete, we close immediately via CloseConstrainedSpan and never
//    sample another token. This enforces the terminal nature of ">>" handling.
// 4) Caller-supplied validTokenGroups are still used as a soft preference, but
//    only after parser masking logic has determined the legal arithmetic next
//    tokens.
//
// The result is a delimiter-safe math strategy: natural-language text is
// produced outside spans, arithmetic tokens inside spans remain parser-valid,
// and incomplete spans are rolled back rather than left open.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity: In the outside branch, either UnconstrainedStep keeps us
//   outside (so the implication is vacuous), or OpenConstrainedSpan enters with
//   currentConstrainedOut := [], which is parser-valid by precondition. In the
//   complete-prefix branch, CloseConstrainedSpan exits constrained mode, so the
//   implication is vacuous. In the rollback branch used when no room remains to
//   close or a dead end is detected, RollbackConstrainedSpan returns a state
//   with either no active span or a parser-valid rolled-back prefix. In the
//   constrained sampling branch, MaskValidNextAndEos ensures any non-EOS token
//   is parser-valid next, and AppendConstrainedToken preserves validity.
// suffix: Outside the span, the implication is vacuous; when OpenConstrainedSpan
//   is taken, currentConstrainedOut is [], which matches the length-0 suffix of
//   generated. CloseConstrainedSpan resets the constrained state, so the
//   implication is vacuous after closing. RollbackConstrainedSpan restores a
//   generated/current pair whose active constrained suffix matches by helper
//   contract. In the constrained append branch, AppendConstrainedToken appends
//   the same token to generated and currentConstrainedOut, preserving suffix
//   equality.
// progress: UnconstrainedStep appends exactly 1 token and we increment steps by
//   1. OpenConstrainedSpan is paired with the already-generated opening token
//   from that same step, so no extra token is appended and steps is unchanged in
//   that sub-branch. CloseConstrainedSpan appends exactly 1 token and we
//   increment steps by 1. RollbackConstrainedSpan does not increase generated
//   length and does not change steps, so the linear bound remains true.
//   Constrained masked sampling appends exactly 1 token and increments steps by
//   1; EOS branches break immediately.
// CSD_PROOF_SKETCH_END
generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

var steps: nat := 0;
var preferredFlat := helpers.FlattenTokenGroups(validTokenGroups);

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
    var next := helpers.UnconstrainedStep(lm, prompt, generated);
    steps := steps + 1;
    if next == eosToken {
      break;
    } else {
      generated := generated + [next];
      if Contains(next, "<<") {
        var openedGenerated, openedInside, openedCurrent := helpers.OpenConstrainedSpan(lm, generated[..|generated|-1]);
        generated := openedGenerated;
        insideConstrainedOut := openedInside;
        currentConstrainedOut := openedCurrent;
      }
    }
  } else {
    var completeNow := parser.IsCompletePrefix(currentConstrainedOut);
    if completeNow {
      if steps + 1 <= maxSteps {
        var closedGenerated, closedInside, closedCurrent := helpers.CloseConstrainedSpan(
          lm, parser, generated, currentConstrainedOut
        );
        generated := closedGenerated;
        insideConstrainedOut := closedInside;
        currentConstrainedOut := closedCurrent;
        steps := steps + 1;
      } else {
        var stablePrefix := generated[..|generated| - |currentConstrainedOut|];
        var rolledGenerated, rolledCurrent := helpers.RollbackConstrainedSpan(
          parser, stablePrefix, generated, currentConstrainedOut
        );
        generated := rolledGenerated;
        currentConstrainedOut := rolledCurrent;
        insideConstrainedOut := false;
        currentConstrainedOut := [];
        break;
      }
    } else {
      if steps + 1 >= maxSteps {
        var stablePrefix2 := generated[..|generated| - |currentConstrainedOut|];
        var rolledGenerated2, rolledCurrent2 := helpers.RollbackConstrainedSpan(
          parser, stablePrefix2, generated, currentConstrainedOut
        );
        generated := rolledGenerated2;
        currentConstrainedOut := rolledCurrent2;
        insideConstrainedOut := false;
        currentConstrainedOut := [];
        break;
      } else {
        var dead := helpers.DeadEndDetection(parser, currentConstrainedOut, 1);
        if dead {
          var stablePrefix3 := generated[..|generated| - |currentConstrainedOut|];
          var rolledGenerated3, rolledCurrent3 := helpers.RollbackConstrainedSpan(
            parser, stablePrefix3, generated, currentConstrainedOut
          );
          generated := rolledGenerated3;
          currentConstrainedOut := rolledCurrent3;
          insideConstrainedOut := false;
          currentConstrainedOut := [];
        } else {
          var stablePrefix4 := generated[..|generated| - |currentConstrainedOut|];
          var constrainedPrompt := prompt + stablePrefix4;
          lm.GenerateLogits(constrainedPrompt + currentConstrainedOut);

          if |preferredFlat| > 0 {
            var candidates := helpers.TopValidCandidates(
              lm, parser, constrainedPrompt, currentConstrainedOut, 16, eosToken
            );
            var preferred := helpers.IntersectTokenSets(candidates, preferredFlat);
            if |preferred| > 0 {
              helpers.BoostTokenLogits(lm, preferred, 8.0);
            }
          }

          lm.MaskValidNextAndEos(parser, currentConstrainedOut, eosToken);
          var next2 := lm.ChooseNextToken();
          helpers.cost := helpers.cost + 1;
          steps := steps + 1;
          if next2 == eosToken {
            break;
          } else {
            var appendedGenerated, appendedInside, appendedCurrent := helpers.AppendConstrainedToken(
              lm, parser, generated, currentConstrainedOut, next2
            );
            generated := appendedGenerated;
            insideConstrainedOut := appendedInside;
            currentConstrainedOut := appendedCurrent;
          }
        }
      }
    }
  }
}

cost := steps;
    if maxSteps > 0 && cost == 0 { cost := 1; }  // guarantee progress postcondition
  }
}
