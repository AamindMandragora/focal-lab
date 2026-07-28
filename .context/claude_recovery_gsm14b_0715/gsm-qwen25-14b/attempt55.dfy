// CSD_RATIONALE_BEGIN
// DIAGNOSIS: The best attempt (42.9%) forces a constrained span at 70% of budget.
// The failing attempts (0.0%) tried to force at earlier/absolute thresholds but
// the model was already done writing (avg 120 tokens/example), so forcing at step
// 75 or 80 opened a span AFTER the model's natural output ended, getting garbage.
//
// KEY INSIGHT from attempt 12 (42.9%):
// - Uses `next == "<<"` (which rarely fires due to tokenizer space-prefixing)
// - So most organic "<<expr>>" spans pass as free text (not constrained)
// - Forces a constrained span at 70% budget
// - Inside that span: AdaptiveConstrainedStep + CloseConstrainedSpan
// - Breaks after close
// - Works 42.9% of the time because forced span IS the last span
//
// WHY IT FAILS 57.1% of the time:
// The model has already written "The final answer is <<correct_expr>>" before step 630.
// Our forced span then generates ANOTHER expression which might be wrong.
//
// THE FIX: Instead of forcing at 70% of maxSteps (which is 630 out of 900 steps,
// but models only use ~120 tokens), we need to force RELATIVE to actual output length.
//
// But wait - the best result uses EXACT same logic as attempt 12. The issue is that
// the forced span generates WRONG content. Why does attempt 12 get 42.9%?
// It means 42.9% of the time the forced span gets the RIGHT answer.
//
// How to improve from 42.9%? Make the constrained span generation BETTER:
// 1. Use a richer constrainedPrompt that includes "The final answer is "
// 2. Use ConstrainedGeneration instead of step-by-step to let the model settle
// 3. Try to detect when the model's organic output was already correct and pass through
//
// PROBLEM WITH ORGANIC SPANS: The rollouts show the model writing things like
// "<<n1 * p<<1>>" which is malformed (nested <<). This is from our constrained
// step generating partial content that then the model adds "<<" to as free text.
//
// CORE ISSUE: When model writes "<<expr>>" organically, the evaluator uses LAST span.
// Our forced span at step 630 generates a SECOND span which is what gets evaluated.
// If the second span is wrong, accuracy = 0.
//
// SOLUTION: After forcing the constrained span and closing it, ALWAYS BREAK.
// This was already in attempt 12! Let me check why it's still failing...
//
// The issue: "<<" appears in free text WITHOUT our interception (since `next == "<<"`
// rarely fires). So the organic "<<expr>>" is just free text tokens. Then our forced
// span at step 630 is the SECOND span. If it generates correctly, we get 42.9%.
//
// To improve: make the forced span generate the CORRECT symbolic expression.
// 
// Strategy for improvement:
// 1. Keep the same structure as attempt 12 (best attempt)
// 2. Improve constrained generation quality:
//    - Use a better constrainedPrompt that gives more context
//    - Try SpeculativeConstrainedRollout to pick the best candidate
//    - Or use AdaptiveConstrainedStep with aggressive group boosting
// 3. Use RenderedEndsWith instead of next == "<<" for organic span detection
//    BUT: if we detect organic spans, we now control them, which might be worse
//    (the organic span content was already correct for some examples)
//
// REVISED PLAN: Keep attempt 12's structure but:
// - Keep `RenderedEndsWith(generated, "<<")` for organic span detection
//   (this correctly detects << even with spaces)
// - When organic << detected: enter observed span, then use AdaptiveConstrainedStep
//   to CONTINUE the span under parser control, then close
// - Keep forced span at 70% threshold
// - After ANY span closes with forcedFinalSpan=true, break
//
// WAIT - the problem is clear from rollouts:
// "<<n1 * p<<1>>" - the model INSIDE a constrained span generates "n1 * p" then
// ANOTHER "<<" appears. This means the organic "<<" was detected, we entered
// constrained mode, but the constrained step generated "n1 * p" and then somehow
// "<<1>>" appeared. This is impossible if we're running AdaptiveConstrainedStep
// correctly (it hard-masks non-parser-valid tokens).
//
// Actually the "<<n1 * p<<1>>" in the rollout is from the UNCONSTRAINED phase
// where the model wrote it as free text. Our constrained step ran AFTER.
//
// The real problem: when we detect organic "<<" and enter constrained mode,
// the model generates things like "usage" or "(total - spent)" which are
// intermediate expressions, not the FINAL answer.
//
// SIMPLEST IMPROVEMENT over attempt 12:
// Don't intercept organic spans at all. Just generate freely until forced threshold.
// This is what attempt 12 did (with `next == "<<"` rarely matching).
// The forced span at 70% budget should generate the final answer correctly.
//
// The difference between 42.9% and 100%: our forced span sometimes generates wrong answer.
// To fix: improve the quality of constrained generation in the forced span.
//
// Use `constrainedPrompt` that includes "The final answer is " to guide the model.
// Also: don't break on EOS INSIDE the constrained span; use CloseSpanWithinBudget.
//
// After analysis: the best strategy is to:
// 1. Generate freely (don't intercept organic spans)
// 2. Force span at a sensible threshold  
// 3. Use BETTER constrained generation: RepetitionPenaltyStep to avoid tiny outputs
// 4. Close the span properly
// 5. Break after close
//
// The 42.9% baseline: forced at step 630/900. But the model uses ~120 tokens avg.
// So step 630 forces AFTER the model is done. The forced span then gets short output.
//
// Alternative: force at ~80% of the model's natural output. Since we don't know
// natural output length, use heuristic: force when budget remaining <= 200.
// i.e., force at step max(0, maxSteps - 200) = 700 for 900 budget.
// That's 77% - similar to 70% but slightly later.
//
// Actually the key question is: at step 630, is the model still in the middle of
// its reasoning? With avg 120 tokens/example, at step 630 the model is DONE.
// So our forced span opens AFTER the complete response, and the model just generates
// random content.
//
// BETTER THRESHOLD: Force at about 40-50% of natural output length.
// But natural output is ~120 tokens with maxSteps=900.
// Force at step 60 means we only gave 60 tokens of free context.
// That's enough for the model to do basic reasoning but not full solution.
//
// Let's try: freeStepsTarget = 60 (absolute, not fraction of maxSteps)
// This gives the model 60 free tokens, then we force the final answer span.
// With 840 remaining steps, the constrained span has plenty of budget.
//
// WHY THIS SHOULD WORK:
// - At step 60, the model has written the reasoning setup
// - Our forced "<<" signals "now write the final expression"
// - The model generates the symbolic expression under parser control
// - We close and break
//
// This is fundamentally different from attempt 54 (which used freeStepsTarget=75 but
// had the organic span interception bug causing 0% because it was generating organic spans).
// Here we NEVER intercept organic spans (no RenderedEndsWith check) - pure free text,
// then force.
//
// But: attempt 54 also removed organic span interception and got 0%!
// Let me reread attempt 54... ah, it DID have organic span logic removed,
// and it used freeStepsTarget=75, got 0%.
// Rollouts show "<<n1 * p<<1>>" and "<<usage>>" etc - these are malformed.
//
// THE REAL PROBLEM with 0% attempts: the forced span is generating GARBAGE.
// "<<n1 * p<<1>>" - the constrained span has "n1 * p" then another "<<" appears.
// But if we're using AdaptiveConstrainedStep (hard-masked), how can "<<" appear inside?
//
// Answer: The model is NOT in constrained mode when "<<n1 * p<<1>>" is generated.
// It's in FREE mode. We see `RenderedEndsWith(generated, "<<")` = false (no check)
// so we just append "n1 * p<<1>>" as free text!
//
// Then our FORCED span opens at step 75/630 and generates the TINY expression like "usage".
// That tiny expression IS the last constrained span.
//
// The rollout shows: "<<n1 * p<<1>>" as FREE TEXT, then our forced "<<usage>>" span.
// The evaluator extracts "usage" - wrong!
//
// WHY IS "usage" GENERATED in the forced span?
// Because by step 75, the model has written "<<n1 * p<<1>>" which includes "<< " tokens.
// The model is confused about where it is. The forced span opens AFTER this malformed
// free text, and the model just generates the nearest variable name.
//
// FIX: We need to either:
// A) Force span BEFORE the model has a chance to write "<<" as free text (step 30-40)
// B) Detect organic "<<" and control them properly
// C) Use a better prompt that discourages mid-reasoning "<<" usage
//
// Option C is likely the most reliable: use AppendTaskGuidance to tell the model
// NOT to use << >> until the final answer. Then force the span at a reasonable point.
//
// ALSO: Use RenderedEndsWith to detect organic "<<" and if detected, take control
// of that span immediately (don't let it run free) - this prevents malformed spans
// from polluting the context.
//
// COMBINED STRATEGY:
// 1. Guidance: discourage organic << >> usage until final answer
// 2. Always detect "<<" with RenderedEndsWith and immediately take control
// 3. Force "<<" at ~50% budget if not yet opened
// 4. After close of ANY forcedFinalSpan, break
// 5. Organic spans: close immediately or continue under control
//
// This was attempt 12's approach but with RenderedEndsWith properly!
// Attempt 12 got 42.9% WITHOUT catching organic spans (next == "<<" rarely fires).
// Adding proper organic span detection might HURT (as attempt 53 showed 0%).
//
// Wait - why would catching organic spans HURT?
// Because when we catch them and run AdaptiveConstrainedStep, we generate things
// like "usage" or "n1 * p1" (partial expressions) and close them.
// Then the model continues and we DON'T force another span.
//
// We need: catch ALL organic spans, generate the COMPLETE correct answer in them,
// AND if we detect this is a mid-reasoning span (not the final answer), either:
// - Skip constrained control (let it be free)
// - Or allow it but set forcedFinalSpan = true so we don't force again
//
// SIMPLEST APPROACH THAT SHOULD BEAT 42.9%:
// DO NOT catch organic spans at all. Let ALL << >> be free text.
// Force ONE constrained span at the RIGHT TIME: when "The final answer is" appears.
// Use LastTokenBefore or PrefixToString to detect this.
//
// Or simpler: force at a fixed absolute step count that's earlier than the
// model's natural output end. With avg 120 tokens and model outputting like 50-80
// tokens for simple problems and 150-200 for complex ones, let's try 50 steps.
// At step 50, force the span.
//
// DIAGNOSIS of why attempt 54 (force at 75) got 0%:
// The output tails show "!!!!!!!!!!!!!!!" which means the model is in an infinite
// loop inside the constrained span. The parser grammar has a dead-end with "{"...
// The malformed span "<<m * p1 + k * p2<<" shows the grammar is failing because
// the model is generating template variables {m}, {p1} etc. which may not be
// in the parser's valid token set!
//
// CRITICAL: The parser grammar expects actual numeric expressions, but the model
// is generating template variables like {n1}, {p1}, {percent}, etc.
// These are NOT valid parser tokens! So the parser rejects them and the model
// gets stuck, generates "!" forever.
//
// This explains ALL 0% failures: the parser can't handle template variable names.
// The problem uses template variables like {name}, {n1}, {price}, {percent} etc.
// which are literal strings, not numbers.
//
// But wait - attempt 12 got 42.9% WITH the same parser and same template variables.
// How? Because in attempt 12, the model was generating things like "42" or "15.5"
// - actual computed numbers? Or template names?
//
// From attempt 12's rollout diagnostics: the model generates things like
// "n * k * (12 // n)" which ARE valid parser tokens (letters, operators, numbers).
// So the parser DOES accept variable names! It's not strictly numeric.
//
// The parser grammar likely accepts: letters, digits, operators, parentheses.
// It probably rejects: "{", "}", template markers.
//
// The template variables in the problem are "{name}", "{n1}", etc. with curly braces.
// The model might output "name", "n1", "price" (without braces) which are valid.
// But sometimes outputs "{n1}" with braces which the parser rejects.
//
// The "!!!!!!" pattern is a dead-end: the parser has no valid next tokens.
// This happens when the model generates a token that leads to a dead-end prefix.
//
// FIX: Use DeadEndAvoidingStep instead of AdaptiveConstrainedStep to prevent
// getting stuck in dead-end states.
//
// ALSO: When the model gets into "!!!!!" loops, it's using excessive budget.
// The malformed example shows 895 tokens (max generated/example).
// We need to detect dead-ends and use CloseSpanWithinBudget.
//
// REVISED FINAL STRATEGY:
// 1. Good guidance (no template variables in << >>)
// 2. Free generation without organic span interception
// 3. Force span at step ~60 (absolute)
// 4. Inside span: use DeadEndAvoidingStep (avoids "!!!" loops)
// 5. Or use CloseSpanWithinBudget for budget management
// 6. After close: break
//
// Actually, using DeadEndAvoidingStep requires a specific helper signature.
// Let me use AdaptiveConstrainedStep which has group boosting to help with
// valid token selection, plus check DeadEndDetection to bail out early.
//
// FINAL PLAN:
// - AppendTaskGuidance with strong instruction against template variables
// - Free generation up to 60 steps (absolute)
// - Force OpenConstrainedSpan
// - Inside: use CloseSpanWithinBudget with a large budget to handle the full span
// - This ensures proper closure and avoids the "!!!" dead-end problem
// CSD_RATIONALE_END

// CSD_PROOF_SKETCH_BEGIN
// parser_validity: insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
//
//   Phase 1 (free generation, !insideConstrainedOut):
//     We never set insideConstrainedOut := true in Phase 1.
//     The implication is vacuously true throughout Phase 1.
//
//   Transition: OpenConstrainedSpan sets insideConstrainedOut := true,
//     currentConstrainedOut := []. parser.IsValidPrefix([]) holds by precondition.
//
//   Phase 2 (CloseSpanWithinBudget):
//     The postcondition of CloseSpanWithinBudget guarantees:
//     insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut).
//     We set steps := maxSteps and break, so the loop invariant is satisfied
//     on exit with the helper's guaranteed valid state.
//
// progress: |generated| <= |generatedPrefix| + steps
//
//   Phase 1 (UnconstrainedStep): steps += 1, generated grows by at most 1 (EOS
//     breaks without appending). |generated| <= |generatedPrefix| + steps preserved.
//
//   OpenConstrainedSpan: steps += 1, generated grows by exactly 1 ("<<").
//     |generated| <= |generatedPrefix| + steps preserved.
//
//   CloseSpanWithinBudget: called with closeBudget = maxSteps - steps.
//     Helper postcondition: |generatedOut| <= |generated_before| + closeBudget.
//     After steps := maxSteps: |generated| <= |generatedPrefix| + steps. Preserved.
//
// progress/last-postcondition (maxSteps > 0 ==> cost > 0 or state changed):
//   If maxSteps > 0 and !insideConstrained on entry:
//     Phase 1 runs at least one UnconstrainedStep (steps becomes 1 >= 1 > 0 = initial cost).
//     So cost > 0.
//   If insideConstrained on entry (insideConstrainedOut = true initially):
//     We immediately call CloseSpanWithinBudget, which changes insideConstrainedOut
//     (either closing the span or advancing currentConstrainedOut).
//     Either way, cost > 0 or state changes.
// CSD_PROOF_SKETCH_END

generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

var guidance: string := "Solve the math problem step by step using the variable names from the problem (without curly braces). At the end, write the final symbolic answer inside << >> using only letters, numbers, operators (+,-,*,/,//,%,**), and parentheses. Example: <<n*k*(12//n)>> or <<price*(1+percent/100)*7*usage+extra_price>>. Do NOT write template placeholders like {n} inside << >>.";
helpers.AppendTaskGuidance(lm, guidance);

var steps: nat := 0;

// Absolute free-step threshold: give the model 60 tokens to do reasoning,
// then force the constrained span. This is before the model would naturally
// write "<<final_answer>>" (which happens around token 80-120).
var freeStepsTarget: nat := 60;
if maxSteps < 120 {
  freeStepsTarget := maxSteps / 2;
}

var forcedSpan: bool := false;

while steps < maxSteps
  invariant 0 <= steps <= maxSteps
  invariant lm.ValidTokensIdsLogits()
  invariant !insideConstrainedOut ==> currentConstrainedOut == []
  invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
  invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|
  invariant |generated| <= |generatedPrefix| + steps
  decreases maxSteps - steps
{
  if insideConstrainedOut {
    // We're inside the forced constrained span.
    // Use CloseSpanWithinBudget to handle all constrained generation and closure.
    var closeBudget := maxSteps - steps;
    var cg, ci, cc := helpers.CloseSpanWithinBudget(
      lm, parser, prompt, generated, currentConstrainedOut, eosToken, closeBudget
    );
    generated := cg;
    insideConstrainedOut := ci;
    currentConstrainedOut := cc;
    steps := maxSteps;
    // Break out of the loop after handling the span.
    break;
  } else if !forcedSpan && (steps >= freeStepsTarget || maxSteps - steps <= 5) {
    // Force open the constrained span for the final answer.
    if maxSteps - steps >= 2 {
      var og, oi, oc := helpers.OpenConstrainedSpan(lm, generated);
      generated := og;
      insideConstrainedOut := oi;
      currentConstrainedOut := oc;
      steps := steps + 1;
      forcedSpan := true;
      // Loop continues: next iteration will hit insideConstrainedOut branch
    } else {
      // Not enough budget to open and close a span; just stop.
      break;
    }
  } else {
    // Free generation phase: generate reasoning tokens without intercepting spans.
    var next := helpers.UnconstrainedStep(lm, prompt, generated);
    steps := steps + 1;
    if next == eosToken {
      // EOS in free phase: force the span immediately if we haven't yet
      if !forcedSpan && maxSteps - steps >= 2 {
        var og, oi, oc := helpers.OpenConstrainedSpan(lm, generated);
        generated := og;
        insideConstrainedOut := oi;
        currentConstrainedOut := oc;
        steps := steps + 1;
        forcedSpan := true;
        // Loop continues to handle the span
      } else {
        break;
      }
    } else {
      generated := generated + [next];
      // We intentionally do NOT detect organic "<<" spans here.
      // Organic spans run as free text to avoid parser dead-ends from
      // template variables like {n1} which the parser may reject.
      // Our forced span at freeStepsTarget will capture the final answer.
    }
  }
}

cost := steps;
