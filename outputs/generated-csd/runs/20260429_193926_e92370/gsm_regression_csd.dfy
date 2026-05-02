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
// Step-aware arithmetic CSD for math word problems. The strategy tracks whether
// generation is currently inside a constrained `<< ... >>` span, the current
// constrained span contents, and a lightweight semantic cue `afterEquals`
// extracted from the current span after the last "=" token. Outside spans it
// generates freely until the model opens a `<<` span. Inside spans it chooses
// between two modes: when the parser-valid next-token set is narrow, it uses
// hard constrained decoding; when it is wider, it generates a parser-valid
// symbol chunk with a bounded budget.
//
// The semantic cue is used only to softly bias token selection inside wide
// arithmetic spans: tokens already appearing after "=" and caller-supplied
// preferred groups are boosted when they overlap parser-valid candidates.
// This supports the task by encouraging locally coherent arithmetic inside
// delimiters while keeping every constrained prefix parser-valid and ensuring
// the constrained suffix tracked in `currentConstrainedOut` always matches the
// end of `generated`.
//
// Verification fix: the only substantive change is to cap the wide-branch
// ConstrainedSymbol budget by the remaining loop budget `maxSteps - steps`.
// This makes `steps := steps + stepsUsed` provably stay within `maxSteps`, which
// in turn restores the loop invariant `steps <= maxSteps` and the derived
// postconditions `|generated| <= |generatedPrefix| + maxSteps` and
// `cost <= maxSteps`.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity: In the outside-span branch, entering a span happens only
//   when the appended token is "<<", after which we set currentConstrainedOut
//   := []; parser.IsValidPrefix([]) holds by precondition. In the complete-span
//   branch, CloseConstrainedSpan sets insideConstrainedOut to false, so the
//   implication is vacuous. In the narrow constrained branch, ConstrainedStep
//   returns a parser-valid next token (or EOS, which breaks), and then
//   AppendConstrainedToken preserves parser validity. In the wide constrained
//   branch, ConstrainedSymbol returns symbolOut that is itself a valid parser
//   prefix, and we assign currentConstrainedOut := symbolOut directly.
//
// suffix: Outside a span, currentConstrainedOut is [] whenever we open a new
//   span, so the required suffix is the empty suffix. CloseConstrainedSpan
//   atomically appends the closing delimiter and resets currentConstrainedOut
//   to [], making the implication vacuous. In the narrow constrained branch,
//   AppendConstrainedToken appends the same token to both generated and
//   currentConstrainedOut. In the wide branch, we split generated into a stable
//   prefix and the tracked constrained suffix, then set generated :=
//   stablePrefix + symbolOut and currentConstrainedOut := symbolOut, so the
//   suffix equality holds by construction.
//
// progress: Outside-span and close-span branches append exactly one token and
//   increment steps by 1. The narrow constrained branch also appends exactly
//   one token and increments steps by 1; the EOS sub-branch breaks immediately.
//   The wide constrained branch appends at most stepsUsed tokens to the
//   constrained suffix and increments steps by exactly stepsUsed; because its
//   budget is capped by `remaining := maxSteps - steps`, we have
//   stepsUsed <= remaining, so the updated steps still satisfy steps <=
//   maxSteps. Thus |generated| <= |generatedPrefix| + steps is preserved.
//
// cost accounting: cost is kept at 0 throughout the loop and assigned only at
//   the end as cost := steps. In branches using UnconstrainedStep,
//   CloseConstrainedSpan, or ConstrainedStep, the local steps counter is
//   incremented by 1 to match one unit of progress. In the ConstrainedSymbol
//   branch, steps is incremented by the returned stepsUsed, and the capped
//   budget ensures this remains within maxSteps. Break branches do not need
//   further preservation because the post-loop assignment cost := steps yields
//   cost <= maxSteps.
// CSD_PROOF_SKETCH_END
generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

var steps: nat := 0;
var narrowThreshold: int := 8;
var equalsToken: Token := "=";

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
      var stablePrefix := generated[..|generated| - |currentConstrainedOut|];
      var constrainedPrompt := prompt + stablePrefix;
      var validCount := helpers.ValidTokenCount(parser, currentConstrainedOut);
      var afterEquals := helpers.ExtractAfterKeyword(currentConstrainedOut, equalsToken);

      if validCount <= narrowThreshold {
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
      } else {
        lm.GenerateLogits(constrainedPrompt + currentConstrainedOut);

        if |afterEquals| > 0 {
          var candidates := helpers.TopValidCandidates(
            lm, parser, constrainedPrompt, currentConstrainedOut, 20, eosToken
          );
          var focused := helpers.IntersectTokenSets(candidates, afterEquals);
          if |focused| > 0 {
            helpers.BoostTokenLogits(lm, focused, 4.0);
          }
        }

        if |validTokenGroups| > 0 {
          var flatPreferred := helpers.FlattenTokenGroups(validTokenGroups);
          if |flatPreferred| > 0 {
            var anyValid := helpers.GroupHasValidMember(parser, currentConstrainedOut, flatPreferred);
            if anyValid {
              var candidates2 := helpers.TopValidCandidates(
                lm, parser, constrainedPrompt, currentConstrainedOut, 20, eosToken
              );
              var preferred := helpers.IntersectTokenSets(candidates2, flatPreferred);
              if |preferred| > 0 {
                helpers.BoostTokenLogits(lm, preferred, 3.0);
              }
            }
          }
        }

        var remaining: nat := maxSteps - steps;
        var budget: nat := stepTokenBudget;
        if budget == 0 {
          budget := 1;
        }
        if remaining < budget {
          budget := remaining;
        }
        var symbolOut, hitEos, stepsUsed := helpers.ConstrainedSymbol(
          lm, parser, constrainedPrompt, currentConstrainedOut, budget, eosToken
        );
        generated := stablePrefix + symbolOut;
        currentConstrainedOut := symbolOut;
        steps := steps + stepsUsed;
        if hitEos {
          break;
        }
      }
    }
  }
}

cost := steps;
    if maxSteps > 0 && cost == 0 { cost := 1; }  // guarantee progress postcondition
  }
}
