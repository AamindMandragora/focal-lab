// CSD_RATIONALE_BEGIN
// ANALYSIS: Attempt 21 regressed from best (49.0%/89.8%) to 30.6%/77.6% because
// removing `observedSpanThreshold` caused `RenderedEndsWith` to detect "<<" tokens
// appearing mid-reasoning — e.g., the model quoting its guidance: "End with
// <<int(formula)>>" — triggering premature constrained entry and leaving 11 spans
// unclosed when the budget ran out before the constrained generation could finish.
//
// ROOT CAUSE OF 11 UNTERMINATED SPANS:
// The model often writes explanatory text containing "<<..." very early (steps 50–150)
// as part of summarizing the task. With no threshold, `RenderedEndsWith` detected
// these early "<<" tokens, entered constrained mode at step ~100, and then the
// constrained generation could not complete within the remaining budget because
// the model's context was mid-reasoning rather than at the final-answer position.
//
// KEY FIX: MINIMUM PRELUDE THRESHOLD
// Only enter constrained mode via `RenderedEndsWith` after `minPreludeSteps = 150`.
// Natural "<<" placement averages step ~237 (median 180), well above this threshold.
// Guidance-quoting "<<" typically appears within the first 150 steps. This threshold
// is API-compliant (uses `RenderedEndsWith`) while restoring attempt-18 behavior.
//
// ALL ATTEMPT-18 PARAMETERS PRESERVED:
//   - prefixBudget = maxSteps - 100 = 800 (force-open threshold)
//   - maxSpanTokens = 22 (max constrained tokens before forced close)
//   - nearBudgetThreshold = 30 (call CloseSpanWithinBudget when budget <= 30)
//   - penaltyTokens = ["{", "}", "**", "^", "\\"] (block invalid formula chars)
//   - break immediately after first clean constrained span close
//   - EOS inside span → CloseSpanWithinBudget with all remaining budget
//   - AdaptiveConstrainedStepWithPenalties for hard parser control inside span
//
// EXPECTED OUTCOME:
//   - Eliminates 11 unterminated spans → syntax 77.6% → ~89.8%
//   - Accuracy restored to ~49% (attempt-18 baseline)
//   - `RenderedEndsWith` properly handles space-prefixed tokens like " <<"
// CSD_RATIONALE_END

// CSD_PROOF_SKETCH_BEGIN
// Invariant 1: parser_validity
//   insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
//
// Branch A (outside span, steps < minPreludeSteps OR RenderedEndsWith=false, non-EOS):
//   insideConstrainedOut stays false. Implication vacuously true.
//
// Branch B (outside span, steps >= minPreludeSteps, RenderedEndsWith=true, non-EOS):
//   generated := generated + [next]; EnterObservedConstrainedSpan sets
//   insideConstrainedOut := true, currentConstrainedOut := [].
//   parser.IsValidPrefix([]) holds by precondition. Preserved.
//
// Branch C (outside span, EOS):
//   insideConstrainedOut stays false. Implication vacuously true.
//
// Branch D (outside span, force-open at prefixBudget):
//   OpenConstrainedSpan appends "<<", sets insideConstrainedOut := true,
//   currentConstrainedOut := []. parser.IsValidPrefix([]) by precondition. Preserved.
//
// Branch E (inside span, nearBudget or spanTokens >= maxSpanTokens):
//   CloseSpanWithinBudget postcondition guarantees
//   insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut). Preserved.
//
// Branch F (inside span, CloseSpanIfComplete closed=true):
//   insideConstrainedOut set to false by helper. Implication vacuously true.
//
// Branch G (inside span, CloseSpanIfComplete closed=false, AdaptiveConstrained non-EOS):
//   AdaptiveConstrainedStepWithPenalties hard-masks to parser-valid tokens plus EOS
//   and returns a parser-valid next token. AppendConstrainedToken appends next to
//   currentConstrainedOut. IsValidPrefix(currentConstrainedOut + [next]) holds by
//   contract of the constrained step. Preserved.
//
// Branch H (inside span, EOS from AdaptiveConstrained, CloseSpanWithinBudget):
//   CloseSpanWithinBudget postcondition guarantees validity. Preserved.
//
// Invariant 2: |generated| <= |generatedPrefix| + steps
//
// Branch A (UnconstrainedStep, non-EOS, no constrained entry):
//   steps += 1 (for the step cost), generated += 1 (append next).
//   EnterObservedConstrainedSpan (when triggered): 0 cost, generated unchanged. Preserved.
//
// Branch B (UnconstrainedStep, EOS):
//   steps += 1, generated unchanged. Preserved.
//
// Branch C (OpenConstrainedSpan at prefixBudget):
//   Appends exactly 1 token "<<" to generated, steps += 1. Preserved.
//
// Branch D (CloseSpanWithinBudget, nearBudget or maxSpan):
//   closeBudget = remainingBudget = maxSteps - steps_before.
//   |generated| grows by at most closeBudget; steps grows by closeBudget.
//   New |generated| <= |gP| + steps_before + closeBudget = |gP| + maxSteps. Preserved.
//
// Branch E (CloseSpanIfComplete closed=true):
//   Appends at most 1 token ">>" to generated, steps += 1. Preserved.
//
// Branch F (CloseSpanIfComplete closed=false + AdaptiveConstrained + AppendConstrainedToken):
//   CloseSpanIfComplete: steps += 1, no visible append (no-op path).
//   AppendConstrainedToken: 0 cost, generated grows by 1.
//   Net: generated +1, steps +1. Preserved.
//
// Branch G (EOS inside span, CloseSpanWithinBudget):
//   rem = maxSteps - steps_after_close_check. steps grows by rem.
//   |generated| grows by at most rem. Preserved.
// CSD_PROOF_SKETCH_END

generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

var guidance: string := "Solve step by step. After your reasoning, write your final answer as <<int(formula)>> using the problem variable names without curly braces (write n not {n}). Use only +, -, *, /, //, % operators. No ** or ^.";
helpers.AppendTaskGuidance(lm, guidance);

var steps: nat := 0;
var minPreludeSteps: nat := 150;
var prefixBudget: nat := if maxSteps > 200 then maxSteps - 100 else (if maxSteps > 100 then maxSteps - 30 else maxSteps / 2);
var penaltyTokens: seq<Token> := ["{", "}", "**", "^", "\\"];
var spanTokens: nat := 0;
var maxSpanTokens: nat := 22;
var nearBudgetThreshold: nat := 30;

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
    var remainingBudget := maxSteps - steps;
    if steps >= prefixBudget && remainingBudget > 5 {
      // Force open a constrained span for the final answer
      var og, oi, oc := helpers.OpenConstrainedSpan(lm, generated);
      generated := og;
      insideConstrainedOut := oi;
      currentConstrainedOut := oc;
      steps := steps + 1;
      spanTokens := 0;
    } else {
      var next := helpers.UnconstrainedStep(lm, prompt, generated);
      steps := steps + 1;
      if next == eosToken {
        break;
      } else {
        generated := generated + [next];
        // Only enter constrained mode after minPreludeSteps to avoid detecting
        // guidance-quoting "<<" that appears early in the model's reasoning
        // (e.g., "End with <<int(formula)>>"). Natural "<<" averages step ~237.
        if steps >= minPreludeSteps && RenderedEndsWith(generated, "<<") {
          var eg, ei, ec := helpers.EnterObservedConstrainedSpan(lm, generated);
          generated := eg;
          insideConstrainedOut := ei;
          currentConstrainedOut := ec;
          spanTokens := 0;
        }
      }
    }
  } else {
    // Inside constrained span
    var remainingBudget := maxSteps - steps;

    // If near budget end or span too long: force close with all remaining budget
    if remainingBudget <= nearBudgetThreshold || spanTokens >= maxSpanTokens {
      if remainingBudget > 0 {
        var sg, si, sc := helpers.CloseSpanWithinBudget(
          lm, parser, prompt, generated, currentConstrainedOut, eosToken, remainingBudget
        );
        generated := sg;
        insideConstrainedOut := si;
        currentConstrainedOut := sc;
        steps := steps + remainingBudget;
      }
      break;
    }

    // Try to close if complete (costs 1 step)
    var cg, ci, cc, closed := helpers.CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut);
    steps := steps + 1;
    if closed {
      generated := cg;
      insideConstrainedOut := ci;
      currentConstrainedOut := cc;
      break;
    } else {
      // Generate next constrained token with penalties on invalid characters
      var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];
      var next := helpers.AdaptiveConstrainedStepWithPenalties(
        lm, parser, constrainedPrompt, currentConstrainedOut,
        validTokenGroups, 2.0,
        penaltyTokens, 8.0,
        6, eosToken
      );
      if next == eosToken {
        // EOS inside span: force close with all remaining budget
        var rem := maxSteps - steps;
        if rem > 0 {
          var sg2, si2, sc2 := helpers.CloseSpanWithinBudget(
            lm, parser, prompt, generated, currentConstrainedOut, eosToken, rem
          );
          generated := sg2;
          insideConstrainedOut := si2;
          currentConstrainedOut := sc2;
          steps := steps + rem;
        }
        break;
      } else {
        var ag, ai, ac := helpers.AppendConstrainedToken(
          lm, parser, generated, currentConstrainedOut, next
        );
        generated := ag;
        insideConstrainedOut := ai;
        currentConstrainedOut := ac;
        spanTokens := spanTokens + 1;
      }
    }
  }
}

cost := steps;
