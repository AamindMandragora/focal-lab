// CSD_RATIONALE_BEGIN
// Root cause analysis from attempt 32 (18.4%) vs best attempt 12 (42.9%):
//
// The core problem is clear from the rollouts:
// - The model generates symbolic template expressions like <<n1 * c1>> organically
// - Our strategy catches these with EnterObservedConstrainedSpan and constrains them
// - But the parser constraint allows math expressions, so <<n1 * c1>> passes validation
// - The evaluator extracts the LAST visible span as the answer
// - The model's final span often contains template variables, not actual numbers
//
// Key insight from rollouts:
// 1. The model generates MANY intermediate spans (avg 11.57 spans/example in attempt 32)
// 2. Each intermediate span contains symbolic math (correct formula, wrong format for extraction)
// 3. The LAST span is what gets extracted as the answer
// 4. With 70% forcing threshold, the forced span fires at step ~630, but the model
//    has ALREADY written "The final answer is <<..." organically at step ~200
// 5. So the forced span fires AFTER the model's reasoning is done, adding extra spans
//    that become the "last visible span" with wrong content
//
// The 42.9% accuracy in attempt 12 came from:
// - Organic span tracking (EnterObservedConstrainedSpan)  
// - The 70% forcing threshold rarely fires (model EOS at ~216 tokens)
// - When it fires, it adds a final constrained answer span
//
// The KEY issue: when the model writes its final "<<answer>>" organically at ~200 steps,
// our constrained generation inside that span produces template variables because:
// a) The parser allows identifier expressions (n1, c1, etc.)
// b) The model's logits at that point want to write template variables
// c) We need to GUIDE the model to write actual computed values
//
// But wait - the diagnostics say "tiny_span_dominant: yes (79%)" for wrong answers.
// This means the constrained spans are tiny (1-3 tokens). If the model organically
// writes <<n1>>, our constrained step allows "n1" as a valid expression token.
//
// The real fix: we should NOT enter constrained mode for organic "<<" spans that
// appear BEFORE the model is ready to give the final answer. Instead, let the model
// generate all its intermediate reasoning (including organic << >> pairs) completely
// unconstrained, THEN force our own constrained span at the end.
//
// Strategy:
// 1. Generate FULLY unconstrained until EOS or very close to budget
// 2. After EOS (or near budget), force a FINAL constrained span
// 3. Use strong constrained generation inside that forced span
// 4. The forced span is AFTER all the model's reasoning, so model logits want to output a number
//
// The key difference from attempt 12: 
// - Don't use EnterObservedConstrainedSpan for organic spans at all
// - Let ALL organic spans be fully unconstrained (the model writes template vars there anyway)
// - Only use constrained generation for our SINGLE forced final span
// - Force the span AFTER EOS is detected (not at 70% of budget)
//
// When to force:
// - If EOS is seen: force span right after
// - If budget is running low (< 30 steps remaining): force span
// - Otherwise: keep generating freely
//
// This way:
// - The model writes its full reasoning including intermediate spans unconstrained
// - At EOS or near-budget, we force a clean final << >> span
// - At that point model context is complete and it should output the actual number
// - The forced span is parser-controlled, ensuring only valid math expressions
//
// The freeStepsTarget in attempt 12 was (maxSteps * 7) / 10 = 630 steps.
// Model typically EOS at ~216-326 tokens (median 291 from diagnostics).
// So the 70% threshold rarely fires, but EOS detection then the forced span will fire.
//
// Updated strategy:
// 1. Free generation (no organic span tracking)
// 2. When EOS seen OR steps > 85% of budget: force final span
// 3. Inside forced span: AdaptiveConstrainedStep with strong guidance
// 4. Close when complete, handle budget pressure with CloseSpanWithinBudget
//
// This should avoid the intermediate span pollution problem.
// The guidance also needs to tell the model to output a FINAL NUMBER at the very end.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity (insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)):
//
//   Branch: UnconstrainedStep outside span, non-EOS, non-"<<":
//     insideConstrainedOut remains false. Implication vacuously true. Preserved.
//
//   Branch: UnconstrainedStep returns EOS or freeStepsTarget reached, no span open:
//     We set eosTriggeredForce := true or break, insideConstrainedOut still false.
//     Implication vacuously true. Preserved.
//
//   Branch: OpenConstrainedSpan forced:
//     Sets insideConstrainedOut := true, currentConstrainedOut := [].
//     parser.IsValidPrefix([]) holds by method precondition. Preserved.
//
//   Branch: CloseConstrainedSpan inside span:
//     Sets insideConstrainedOut := false. Implication vacuously true. Preserved.
//
//   Branch: CloseSpanWithinBudget inside span:
//     Postcondition: insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut). Preserved.
//
//   Branch: AdaptiveConstrainedStep non-EOS + AppendConstrainedToken:
//     AdaptiveConstrainedStep hard-masks to valid next tokens, so next preserves
//     parser.IsValidPrefix(currentConstrainedOut + [next]). AppendConstrainedToken
//     extends currentConstrainedOut by that token. Preserved.
//
//   Branch: AdaptiveConstrainedStep returns EOS:
//     We break or use CloseSpanWithinBudget; both paths preserve the invariant. Preserved.
//
// progress (|generated| <= |generatedPrefix| + steps):
//
//   UnconstrainedStep: steps += 1, generated grows by at most 1 (EOS: no append, break).
//     |generated| <= |generatedPrefix| + steps. Preserved.
//
//   OpenConstrainedSpan: steps += 1, generated grows by exactly 1 ("<<").
//     |generated| <= |generatedPrefix| + steps. Preserved.
//
//   CloseConstrainedSpan: steps += 1, generated grows by at most 1 (">>").
//     |generated| <= |generatedPrefix| + steps. Preserved.
//
//   CloseSpanWithinBudget with closeBudget = maxSteps - steps:
//     Postcondition: |generatedOut| <= |generated| + closeBudget = |generated| + maxSteps - steps.
//     After steps := maxSteps: |generated| <= |generatedPrefix| + maxSteps = |generatedPrefix| + steps. Preserved.
//
//   AdaptiveConstrainedStep + AppendConstrainedToken: steps += 1, AppendConstrainedToken
//     appends exactly 1 token. |generated| <= |generatedPrefix| + steps. Preserved.
// CSD_PROOF_SKETCH_END
generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

// Guidance: model should finish its reasoning and output a single number at the end.
var guidance: string := "Solve step by step with the specific numeric values given. After showing your work, write ONLY the final numeric answer (a single integer or decimal) inside << >> delimiters at the very end. Example: <<42>>. Do not write symbolic expressions or variable names in the final << >>.";
helpers.AppendTaskGuidance(lm, guidance);

var steps: nat := 0;
var narrowThreshold: nat := 12;
// Force a constrained span after 88% of budget (safety net for non-EOS cases)
var freeStepsTarget: nat := (maxSteps * 88) / 100;
var forcedFinalSpan: bool := false;
// Track whether we should force the span (set when EOS seen during free gen)
var shouldTriggerForcedSpan: bool := insideConstrained;

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
    var remainingSteps := maxSteps - steps;
    // Force a span when:
    // 1. EOS was seen during free gen (model finished reasoning) OR
    // 2. Budget pressure (>88% used) OR
    // 3. Very low budget (< 10 steps left)
    var budgetPressure := steps >= freeStepsTarget || remainingSteps <= 10;
    var shouldForce := !forcedFinalSpan && (shouldTriggerForcedSpan || budgetPressure);
    
    if shouldForce && remainingSteps >= 3 {
      // Force open a constrained span for the final numeric answer
      var og, oi, oc := helpers.OpenConstrainedSpan(lm, generated);
      generated := og;
      insideConstrainedOut := oi;
      currentConstrainedOut := oc;
      steps := steps + 1;
      forcedFinalSpan := true;
      shouldTriggerForcedSpan := false;
    } else if shouldForce && remainingSteps < 3 {
      // Not enough budget to open and close a span - stop
      break;
    } else {
      // Free unconstrained generation (model writes its reasoning with organic spans)
      var next := helpers.UnconstrainedStep(lm, prompt, generated);
      steps := steps + 1;
      if next == eosToken {
        // Model finished - trigger forced span on next iteration
        shouldTriggerForcedSpan := true;
        // Don't break - let the loop handle forcing the span next iteration
      } else {
        generated := generated + [next];
        // Do NOT enter constrained mode for organic "<<" spans.
        // Let them be completely unconstrained - they contain intermediate formulas.
        // We only constrain our single forced final span at the end.
      }
    }
  } else {
    // Inside the forced constrained span - generate the final numeric answer
    if parser.IsCompletePrefix(currentConstrainedOut) {
      // Span has valid complete content - close it
      var cg, ci, cc := helpers.CloseConstrainedSpan(
        lm, parser, generated, currentConstrainedOut
      );
      generated := cg;
      insideConstrainedOut := ci;
      currentConstrainedOut := cc;
      steps := steps + 1;
      // After closing the final forced span, we're done
      break;
    } else if maxSteps - steps <= 5 {
      // Near budget end: use CloseSpanWithinBudget to finish gracefully
      var closeBudget := maxSteps - steps;
      var cg, ci, cc := helpers.CloseSpanWithinBudget(
        lm, parser, prompt, generated, currentConstrainedOut, eosToken, closeBudget
      );
      generated := cg;
      insideConstrainedOut := ci;
      currentConstrainedOut := cc;
      steps := maxSteps;
    } else {
      // Generate next token of the final answer under parser control
      var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];
      var next := helpers.AdaptiveConstrainedStep(
        lm, parser, constrainedPrompt, currentConstrainedOut,
        validTokenGroups, 4.0, narrowThreshold, eosToken
      );
      steps := steps + 1;
      if next == eosToken {
        // EOS inside constrained span - try to close gracefully
        var remainingAfterEos := maxSteps - steps;
        if remainingAfterEos >= 1 {
          var closeBudget := remainingAfterEos;
          var cg, ci, cc := helpers.CloseSpanWithinBudget(
            lm, parser, prompt, generated, currentConstrainedOut, eosToken, closeBudget
          );
          generated := cg;
          insideConstrainedOut := ci;
          currentConstrainedOut := cc;
          steps := maxSteps;
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

cost := steps;
