// CSD_RATIONALE_BEGIN
// DEEP ANALYSIS OF FAILURE PATTERNS:
//
// Current results (attempt 39): accuracy 18.4%, syntax 87.8%
// Best result (attempt 27): accuracy 26.5%, syntax 85.7%
//
// KEY DIAGNOSTICS FROM ATTEMPT 39:
// - mode_N: 19 examples - syntax_valid, used_constrained=YES, tiny_span_dominant=YES
//   -> The constrained mode runs but produces TINY spans (only 1-2 tokens like "count")
//   -> CloseSpanIfComplete fires immediately on a single variable name
// - mode_B: 13 examples - syntax_valid, used_constrained=NO
//   -> Model writes its own span naturally, correctly formatted, but WRONG formula
// - mode_F: 3 examples - unclosed spans
// - Constrained helper call fraction: 0.01 (1%)
// - "Examples with only tiny valid visible spans: 19/49"
//
// ROOT CAUSE OF mode_N (19 examples):
// The model writes "<<" and enters constrained mode. Then:
// 1. CloseSpanIfComplete is called FIRST in the loop
// 2. If currentConstrainedOut is EMPTY ([]), parser.IsCompletePrefix([]) may be true
//    (if the grammar accepts empty completion)
// OR
// 3. ConstrainedStep generates "n1" (a valid variable), AppendConstrainedToken appends it,
//    then on next iteration CloseSpanIfComplete fires on "n1" which is a complete expression
//    -> span becomes "<<n1>>" which is too tiny
//
// This is the "tiny span" problem. The parser accepts single variable names as complete,
// so CloseSpanIfComplete fires too early.
//
// ROOT CAUSE OF mode_B (13 examples):
// 25 examples have NO constrained activity (examples_without_activity=25).
// The model writes its "<<" at ~204 tokens avg (median 138!).
// At that point, the phase 1 loop (free gen) hasn't reached reasoningBudget (600).
// The model writes "<<" -> we enter constrained mode.
// But then in phase 2, CloseSpanIfComplete fires FIRST on empty/tiny currentConstrainedOut
// -> span closes immediately -> we exit constrained mode with tiny span!
// Then the model continues freely and writes the REST of its expression naturally.
// The evaluator extracts the LAST visible span which is the natural one (mode_B).
//
// WAIT - but mode_B says used_constrained=NO. So why would it fire CloseSpanIfComplete
// and exit if that would count as constrained activity?
//
// Actually "used_constrained" means ConstrainedStep was called. CloseSpanIfComplete alone
// might not count. So:
// 1. Model writes "<<" at token 138 (median)
// 2. We enter insideConstrainedOut=true, currentConstrainedOut=[]
// 3. CloseSpanIfComplete([]) is called -> if parser.IsCompletePrefix([]) then closes!
// 4. generated gets ">>" appended, insideConstrainedOut=false
// 5. We broke out of the constrained loop WITHOUT calling ConstrainedStep
// 6. Mode_B: syntax valid (the model's NATURAL later span), used_constrained=NO
//
// So the problem is: CloseSpanIfComplete fires on EMPTY currentConstrainedOut!
// The empty expression is "complete" according to the parser? Or the span close is forced?
//
// Let me reconsider. Looking at the code in attempt 39:
// - Phase 2 loop starts with CloseSpanIfComplete FIRST
// - If steps + 30 >= maxSteps: break (reserves budget)
// - Then CloseSpanIfComplete: if empty prefix is complete, closes immediately
// - ConstrainedStep runs only if NOT closed
//
// THE FIX: Don't call CloseSpanIfComplete at the START of the loop.
// Instead: ConstrainedStep first, THEN check for close.
// OR: Skip CloseSpanIfComplete entirely; rely only on CloseSpanWithinBudget.
//
// REVISED STRATEGY (combining best elements):
// 1. Free generation until "<<" detected (NO reasoning budget forcing)
//    The model ALWAYS writes "<<" naturally (49/49 examples, avg 204 tokens).
//    Don't force-open if not needed.
// 2. When "<<" detected: enter constrained mode
// 3. Phase 2: ConstrainedStep FIRST, then CloseSpanIfComplete AFTER appending
//    -> This ensures we always append at least 1 constrained token before checking close
//    -> Prevents tiny "<<n>>" spans
// 4. BUT we need to prevent premature close on single var.
//    Add a minConstrainedTokens threshold: don't close until we have >= 2 constrained tokens.
//    Actually better: call CloseSpanIfComplete only AFTER generating >= 3 tokens.
// 5. CloseSpanWithinBudget for fallback.
//
// ABOUT minConstrainedTokens:
// The parser accepts "n1" as complete (it's a valid expression).
// But we want "n1 + n2 * mult" etc. (longer expressions).
// We can't force the parser to NEED more tokens if it accepts the shorter form.
// However, we CAN delay checking completeness until we've accumulated enough tokens.
// If we delay CloseSpanIfComplete until |currentConstrainedOut| >= MIN_TOKENS,
// the model generates a longer expression naturally.
//
// PROBLEM with this: the model might generate an invalid extension after the complete point.
// E.g., "n1" is complete. ConstrainedStep at "n1" might return EOS (since the expression
// is complete). If we ignore EOS and keep going, we loop forever.
//
// But ConstrainedStep returns EOS when the expression is complete. So after "n1":
// next = ConstrainedStep(.., "n1", eosToken) = eosToken (model wants to close)
// We then handle EOS by calling CloseSpanIfComplete.
// This gives us "<<n1>>" - tiny span.
//
// TO PREVENT TINY SPANS: We need to keep generating PAST the point where the parser
// accepts a completion. This requires NOT treating EOS as a signal to close.
// But the model sends EOS when it's done with the expression...
//
// ALTERNATIVE: Use the prompt to TELL the model to write complex expressions.
// Better guidance: "The answer must be a mathematical expression with multiple terms
// using addition, subtraction, multiplication, and/or division."
//
// From the rollout examples:
// - "n - int(n * frac)" - model writes "n - n * frac" (close but missing int())
// - "count*(n1+n2+n3+n4+n5)" - model writes just "count" (way too short!)
// - The "count" case: model writes the correct answer in reasoning but "<<count>>" as span
//
// So tiny spans ARE a real problem. The model writes "<<count>>" when the correct is
// "<<count*(n1+n2+n3+n4+n5)>>".
//
// GUIDANCE KEY INSIGHT: The examples show the model writing "<<" early (median 138).
// This is BEFORE it has reasoned to the full answer. The model hasn't yet figured out
// the expression, so it starts the span with a simple variable and closes.
//
// THE REAL FIX: We need the model to write "<<" AFTER reasoning.
// => We should NOT enter constrained mode on the FIRST "<<".
// => We should enter constrained mode on the LAST "<<" before the answer is finalized.
// => OR: Use a minimum reasoning threshold (e.g., 200 tokens) before detecting "<<".
//
// With median=138 and avg=204, a threshold of 200 would skip the early "<<" in ~50% of cases.
// A threshold of 250 would skip in ~75% of cases.
// A threshold of 350 would skip in ~90%+ of cases.
//
// BUT: the examples show "Examples with visible `<<`: 49/49" - ALL examples have "<<".
// The question is whether the model writes a SECOND "<<" if we skip the first.
// Looking at rollout: "The final answer is <<n1 * frac + n2>>" - this is the ONLY "<<".
// So if we skip it, we might miss the only one.
//
// CONTRADICTORY REQUIREMENTS:
// 1. We need to enter constrained mode (to fix wrong expressions)
// 2. The model's early "<<" produces tiny spans
// 3. The model's free span is semantically wrong in 35/49 cases
//
// PROPOSED SOLUTION: Enter constrained mode on ALL "<<", but DON'T use CloseSpanIfComplete.
// Use ONLY CloseSpanWithinBudget after generating MANY constrained tokens.
// This forces the model to generate a complete expression under parser control.
//
// Specifically:
// - Phase 1: Free gen until "<<" (NO budget forcing - model always writes <<)
// - Phase 2: ConstrainedStep in a loop for up to K constrained tokens
//   NEVER call CloseSpanIfComplete inside the loop
//   Stop when EOS OR budget running low
// - Phase 3: CloseSpanWithinBudget to force close
//
// This ensures:
// - tiny spans: model generates at least some tokens before forced close
// - unclosed spans: CloseSpanWithinBudget handles this
// - malformed spans: ConstrainedStep prevents curly braces etc.
//
// QUESTION: How many constrained tokens to generate before Phase 3?
// The budget is 900. Model uses avg 204 tokens before "<<". Budget after: 696.
// We can afford 200-300 constrained tokens + CloseSpanWithinBudget (50 tokens).
//
// But 696 tokens of constrained generation is WAY too much for a single expression.
// Set max constrained loop to e.g. 50 tokens, then force close.
//
// But if we limit to 50 constrained tokens, the expression could be long:
// "(n1 + n2 + n3) * fraction * (1 - tax) * count + fixed_cost" could be many tokens.
// 50 should be enough for most GSM expressions.
//
// REVISED PLAN:
// - Phase 1: Free generation until "<<" detected OR budget exhausted
//   Budget limit: 600 tokens (to ensure time for Phase 2 + 3)
// - If budget exhausted without "<<": force open span
// - Phase 2: ConstrainedStep loop for at most 80 tokens  
//   EXIT on EOS (natural completion by model)
//   EXIT when budget running low (reserve 50 for Phase 3)
// - Phase 3: CloseSpanWithinBudget with remaining budget
// - NO CloseSpanIfComplete in Phase 2 to avoid tiny spans
//
// GUIDANCE IMPROVEMENT:
// Current guidance says "Write exactly ONE expression". This works.
// Add: "Before writing <<, show your complete calculation. The expression inside << >>
//  should be the full formula with all relevant variables combined."
// This encourages more thorough reasoning and a fuller expression.
//
// KEY CHANGE vs attempt 39: REMOVE CloseSpanIfComplete from Phase 2.
// KEY CHANGE vs best (27): Keep "no CloseSpanIfComplete" approach but also
// DON'T allow the EOS path to close with CloseSpanIfComplete.
// Instead, break out of Phase 2 on EOS and let Phase 3 (CloseSpanWithinBudget) handle it.
//
// RISK: If CloseSpanWithinBudget fails to close (budget=50 too small), span stays open.
// MITIGATION: Give Phase 3 more budget (at least 100 tokens).
// With 900 total, 600 free + small constrained + 100 close = ~750 max, leaving 150 slack.
//
// Actually, let's be more aggressive about the Phase 3 budget.
// Reserve budget = maxSteps * 1 / 3 = 300 tokens for Phase 2 + Phase 3.
// Use Phase 2 for at most 100 tokens.
// Use Phase 3 (CloseSpanWithinBudget) for the remaining 200 tokens.
// This ensures ample budget for closure.
//
// FINAL PLAN:
// - Guidance: emphasize multi-variable complex expression
// - Phase 1: Free gen, reasoningBudget = maxSteps * 2 / 3 (600 tokens)
//   Force open after budget
// - Phase 2: ConstrainedStep, at most maxSteps/6 steps (150 tokens)
//   NO CloseSpanIfComplete. Break on EOS.
//   Reserve 50 steps for Phase 3.
// - Phase 3: CloseSpanWithinBudget with all remaining budget
//
// This should address:
// - mode_N (tiny spans): No CloseSpanIfComplete means we run more ConstrainedStep
// - mode_B (unconstrained wrong): Force reasoning budget ensures we enter constrained
// - mode_F (unclosed): Phase 3 with ample budget
// - malformed content: ConstrainedStep prevents curly braces
//
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity: insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
//
// Phase 1 (free generation loop):
//   - next != "<<" and next != eosToken: generated grows by 1, insideConstrainedOut stays false.
//     Implication is vacuously true (antecedent false).
//   - next == "<<": insideConstrainedOut := true, currentConstrainedOut := [].
//     parser.IsValidPrefix([]) holds by precondition. Invariant satisfied.
//   - Budget exceeded -> OpenConstrainedSpan: sets insideOut=true, currentOut=[].
//     parser.IsValidPrefix([]) holds by precondition. Invariant satisfied.
//   - EOS: break, no state change. Invariant preserved by IH.
//
// Phase 2 (constrained loop, NO CloseSpanIfComplete):
//   - ConstrainedStep returns parser-valid token next != eosToken:
//     AppendConstrainedToken extends currentConstrainedOut with a valid next token.
//     By definition of IsValidPrefix (extending valid prefix with valid next token),
//     parser.IsValidPrefix(currentConstrainedOut + [next]) holds. Invariant preserved.
//   - EOS: break. currentConstrainedOut unchanged (still valid prefix). Invariant preserved.
//   - Budget check (break early): state unchanged. Invariant preserved.
//
// Phase 3 (CloseSpanWithinBudget):
//   Postcondition of CloseSpanWithinBudget guarantees:
//   insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut).
//   So invariant is trivially preserved after phase 3.
//
// progress: |generated| <= |generatedPrefix| + steps
//
// Phase 1:
//   Each iteration: UnconstrainedStep costs +1 step, appends at most 1 token (0 if EOS).
//   OpenConstrainedSpan: +1 step (forced "<<" token), appends exactly 1 token.
//   So after k iterations: |generated| <= |generatedPrefix| + k = |generatedPrefix| + steps. ✓
//
// Phase 2:
//   Each iteration: ConstrainedStep costs +1 step, AppendConstrainedToken appends at most 1 token.
//   EOS path: +1 step, 0 tokens appended. Both: |generated| grows by at most steps. ✓
//
// Phase 3:
//   CloseSpanWithinBudget postcondition: |generatedOut| <= |generated_in| + closeBudget.
//   closeBudget = maxSteps - steps, so |generatedOut| <= |generatedPrefix| + steps + (maxSteps - steps)
//   = |generatedPrefix| + maxSteps. We set steps := maxSteps, so cost = maxSteps. ✓
// CSD_PROOF_SKETCH_END
generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

if maxSteps == 0 {
  cost := 0;
} else {
  var guidance: string := "Solve step by step. Show your full reasoning including all intermediate calculations. At the very end, write your final mathematical expression inside << >> using ONLY plain variable names (no curly braces, no currency symbols) and operators +, -, *, /, //, %. The expression MUST combine ALL relevant numeric variables from the problem with appropriate operators. Do NOT write just a single variable name. Examples: <<count * (n1 + n2 + n3)>>, <<n * price - quantity * discount>>, <<base * rate // 100 + fixed>>. Write exactly ONE expression inside << >>.";
  helpers.AppendTaskGuidance(lm, guidance);

  var steps: nat := 0;
  // Reasoning budget: allow up to 2/3 of maxSteps for free reasoning
  var reasoningBudget: nat := (maxSteps * 2) / 3;

  // Phase 1: Free generation until "<<" appears or reasoning budget exhausted
  while steps < maxSteps && !insideConstrainedOut
    invariant 0 <= steps <= maxSteps
    invariant lm.ValidTokensIdsLogits()
    invariant !insideConstrainedOut ==> currentConstrainedOut == []
    invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
    invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|
    invariant |generated| <= |generatedPrefix| + steps
    decreases maxSteps - steps
  {
    // Check if we've hit the reasoning budget - force span open
    if steps >= reasoningBudget {
      var og, oi, oc := helpers.OpenConstrainedSpan(lm, generated);
      generated := og;
      insideConstrainedOut := oi;
      currentConstrainedOut := oc;
      steps := steps + 1;
      break;
    }

    var next := helpers.UnconstrainedStep(lm, prompt, generated);
    steps := steps + 1;
    if next == eosToken {
      break;
    }
    generated := generated + [next];
    if next == "<<" {
      insideConstrainedOut := true;
      currentConstrainedOut := [];
    }
  }

  // Phase 2: Constrained generation - NO CloseSpanIfComplete to avoid tiny spans
  // Generate constrained tokens until EOS or budget threshold
  // Reserve sufficient budget for Phase 3 (CloseSpanWithinBudget)
  var phase3Reserve: nat := maxSteps / 6;
  if phase3Reserve < 50 { phase3Reserve := 50; }

  while steps < maxSteps && insideConstrainedOut
    invariant 0 <= steps <= maxSteps
    invariant lm.ValidTokensIdsLogits()
    invariant !insideConstrainedOut ==> currentConstrainedOut == []
    invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
    invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|
    invariant |generated| <= |generatedPrefix| + steps
    decreases maxSteps - steps
  {
    // Reserve budget for Phase 3 (CloseSpanWithinBudget)
    if steps + phase3Reserve >= maxSteps {
      break;
    }

    var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];
    var next := helpers.ConstrainedStep(lm, parser, constrainedPrompt, currentConstrainedOut, eosToken);
    steps := steps + 1;

    if next == eosToken {
      // Model wants to end - exit Phase 2, let Phase 3 handle closing
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

  // Phase 3: Force close if still inside span with remaining budget
  if insideConstrainedOut && steps < maxSteps {
    var closeBudget := maxSteps - steps;
    var cg, ci, cc := helpers.CloseSpanWithinBudget(
      lm, parser, prompt, generated, currentConstrainedOut, eosToken, closeBudget
    );
    generated := cg;
    insideConstrainedOut := ci;
    currentConstrainedOut := cc;
    steps := maxSteps;
  }

  cost := steps;
}
