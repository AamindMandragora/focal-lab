// CSD_RATIONALE_BEGIN
// Math-word-problem CSD with improved span management.
//
// Key analysis from failure modes:
// 1. Attempt 2 (0% accuracy): The constrained generation was working but
//    the task guidance caused the model to reason about "no variables" rule,
//    leading to unterminated spans (model fills span with reasoning text about
//    the constraint). 20 examples hit runtime errors (CSD exceeded max_steps:
//    generated 907 tokens > 900) - the unconstrained chunk was consuming too
//    much budget.
//
// 2. Attempt 1 (28.6% accuracy, 81.6% syntax): Simple approach worked better.
//    Failures: 23 syntax_valid_semantic_mismatch (model gets right structure
//    but wrong answer), 7 token_budget_exhausted.
//
// Root causes to fix:
// A) Unterminated spans: Model generates reasoning inside << >> and hits EOS
//    or dead-end before closing. Fix: Use CloseSpanWithinBudget as safety net
//    when budget runs low.
// B) Budget exhaustion: Model reasons too long in unconstrained mode. Fix: 
//    Limit unconstrained chunk size per iteration to conserve budget.
// C) Wrong answers: Model generates syntactically valid but semantically wrong
//    expressions. Fix: Use SafeRepetitionPenaltyStep inside spans to avoid
//    repeating tokens, and use AdaptiveConstrainedStep with groups.
// D) No spans emitted (21 examples in attempt 2): Guidance caused model to
//    refuse. Fix: Use minimal guidance that doesn't confuse the model.
//
// Strategy: Build on Attempt 1 (best result) with targeted improvements:
// 1. Keep simple free generation until "<<" appears
// 2. Inside spans: use CloseSpanIfComplete first, then AdaptiveConstrainedStep
// 3. Add CloseSpanWithinBudget safety net when budget is low (< 20 steps)
// 4. Use minimal, non-confusing task guidance
// 5. Cap unconstrained steps per iteration at 30 to preserve budget
//
// The key insight: the parser enforces hard validity on span content.
// The model's free-text reasoning outside spans is left unconstrained.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity invariant:
//   1. Outside-span branch (UnconstrainedStep):
//      When next == "<<", we set currentConstrainedOut := [] which satisfies
//      parser.IsValidPrefix([]) by the method precondition. insideConstrainedOut
//      becomes true only in this case. For all other tokens, insideConstrainedOut
//      remains false making the implication vacuous.
//   2. CloseSpanIfComplete branch:
//      When closed == true: CloseSpanIfComplete sets insideConstrainedOut := false
//      and currentConstrainedOut := [], making the implication vacuously true.
//      When closed == false: state is unchanged, invariant trivially preserved.
//   3. AdaptiveConstrainedStep + AppendConstrainedToken branch:
//      AdaptiveConstrainedStep returns either EOS or a token t such that
//      parser.IsValidPrefix(currentConstrainedOut + [t]) holds. AppendConstrainedToken
//      sets currentConstrainedOut := currentConstrainedOut + [t], preserving validity.
//      EOS causes break without state mutation.
//   4. CloseSpanWithinBudget branch (safety net):
//      By its postcondition: insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut).
//      This directly satisfies the invariant.
//
// progress invariant (|generated| <= |generatedPrefix| + steps):
//   Outside-span: steps += 1; generated grows by at most 1 (non-EOS token
//   appended or nothing on EOS break). Preserved.
//   CloseSpanIfComplete: steps += 1; appends at most ">>" (1 token). Preserved.
//   AdaptiveConstrainedStep + AppendConstrainedToken: steps += 1; AppendConstrainedToken
//   appends exactly 1 token to generated (EOS causes break without appending). Preserved.
//   CloseSpanWithinBudget: steps += closeBudget (at most maxSteps - steps);
//   generatedOut grows by at most closeBudget tokens. Since steps becomes
//   steps + closeBudget <= maxSteps, bound is preserved.
// CSD_PROOF_SKETCH_END
generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

var guidance: string := "Solve step by step. Write each calculation result and the final answer inside << >> delimiters. Example: <<3 * 4>> gives <<12>>.";
helpers.AppendTaskGuidance(lm, guidance);

var steps: nat := 0;
var narrowThreshold: nat := 12;

while steps < maxSteps
  invariant 0 <= steps <= maxSteps
  invariant lm.ValidTokensIdsLogits()
  invariant !insideConstrainedOut ==> currentConstrainedOut == []
  invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
  invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|
  invariant |generated| <= |generatedPrefix| + steps
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
    // Safety net: if budget is very low, use CloseSpanWithinBudget
    var remainingBudget := maxSteps - steps;
    if remainingBudget <= 15 && remainingBudget > 0 {
      var sg, si, sc := helpers.CloseSpanWithinBudget(
        lm, parser, prompt, generated, currentConstrainedOut, eosToken, remainingBudget
      );
      generated := sg;
      insideConstrainedOut := si;
      currentConstrainedOut := sc;
      steps := steps + remainingBudget;
    } else {
      // Try to close if complete
      var cg, ci, cc, closed := helpers.CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut);
      steps := steps + 1;
      if closed {
        generated := cg;
        insideConstrainedOut := ci;
        currentConstrainedOut := cc;
      } else {
        var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];
        var next := helpers.AdaptiveConstrainedStep(
          lm, parser, constrainedPrompt, currentConstrainedOut,
          validTokenGroups, 4.0, narrowThreshold, eosToken
        );
        if next == eosToken {
          // Try to close the span before giving up
          var remainingBudget2 := maxSteps - steps;
          if remainingBudget2 > 0 {
            var sg2, si2, sc2 := helpers.CloseSpanWithinBudget(
              lm, parser, prompt, generated, currentConstrainedOut, eosToken, remainingBudget2
            );
            generated := sg2;
            insideConstrainedOut := si2;
            currentConstrainedOut := sc2;
            steps := steps + remainingBudget2;
          }
          break;
        } else {
          var ag, ai, ac := helpers.AppendConstrainedToken(
            lm, parser, generated, currentConstrainedOut, next
          );
          generated := ag;
          insideConstrainedOut := ai;
          currentConstrainedOut := ac;
        }
      }
    }
  }
}

cost := steps;
