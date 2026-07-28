// CSD_RATIONALE_BEGIN
//
// ANALYSIS OF CURRENT SITUATION (attempt 39 → 40):
//
// Best result: attempt 36 with 40.8% accuracy, 91.8% syntax.
// Attempt 39: 36.7% accuracy - regression from best.
//
// KEY FAILURE MODES FROM ATTEMPT 39:
// - mode_H: 15 examples (constrained active, wrong answer) 
// - mode_A: 12 examples (natural output, wrong answer)
// - mode_G: 3 examples (unterminated span - STILL OCCURRING despite fix attempt)
// - mode_D: 1 example (invalid constrained content with **)
//
// UNTERMINATED SPAN ANALYSIS (mode_G: 3 examples):
// The output tails show:
// 1. "...the result must be an integer count or monetary amount." If the inputs are integers 
//    and fraction is an<<int fraction!"
//    → Model is MID-SENTENCE when the forced "<<" appears. The model then generates "int" 
//    as the first constrained token, followed by " fraction!" which breaks the grammar.
//    Wait: this can't be from Phase 3 constrained (which should generate valid tokens).
//    Let me re-read: the output tail shows the FULL output including preamble text.
//    The "<<" appears in the middle of the model's reasoning text where it wrote 
//    "...fraction is an<<" as in "an << int(...)". Wait that makes no sense.
//    Actually: "an<<int fraction!" = model wrote "an<<" then our Phase 2 opened constrained,
//    then Phase 3 generated "int" then... "fraction" + "!" failing syntax.
//    No wait: Phase 3 uses ConstrainedStep which would only generate valid tokens.
//    
//    OH! The output shows "...is an<<int fraction!" which suggests the model's OWN output
//    wrote "int fraction!" BEFORE our constrained phase.
//    This means: model wrote "...fraction is an<<int fraction!..." during PREAMBLE.
//    The "<<" is a MID-SENTENCE occurrence in preamble text.
//    Our preamble code does NOT intercept "<<" (pure unconstrained).
//    Then the model DIDN'T hit EOS in preamble (too much text after "<<").
//    So Phase 2 fires and opens "<<" AGAIN at position after "int fraction!" text.
//    But the output ENDS with "an<<int fraction!" suggesting the PREAMBLE ended mid-sentence.
//    
//    REVISED UNDERSTANDING: The "output tail" shows what's in GENERATED after all phases.
//    For mode_G examples (zero_visible_spans): the evaluator found NO valid "<<>>" pair.
//    The "<<" at the very end suggests: the LAST thing in generated is an unclosed "<<".
//    Wait but Phase 4 should close with CloseSpanWithinBudget...
//    
//    ACTUAL BUG: In attempt 39, we have Phase 3b guard: "steps + 20 < maxSteps".
//    But the TAIL shows "...an<<int fraction!" which ends with "fraction!".
//    This "<<" is from OUR Phase 2 OpenConstrainedSpan, and "int", "fraction", "!" are from...
//    Wait: "!" is NOT a valid constrained token. ConstrainedStep wouldn't generate "!".
//    So "int fraction!" must be from the PREAMBLE (unconstrained), not from Phase 3.
//    
//    Let me re-read the output tail: 
//    '...the result must be an integer count or monetary amount." If the inputs are integers 
//    and fraction is an<<int fraction!'
//    
//    This entire text is IN THE PREAMBLE (Phase 1, unconstrained).
//    The model wrote "fraction is an<<int fraction!" as free text.
//    The "<<" here is just part of the model's text (like a comparison operator or tag).
//    After the preamble ends (750 steps), the preamble loop exits.
//    Phase 2 fires: !insideConstrainedOut → OpenConstrainedSpan → appends "<<".
//    But then Phase 3/4 should generate constrained content and close with ">>".
//    WHY IS THE SPAN UNCLOSED?
//
//    BUDGET ISSUE: The preamble ran 750 steps. The model's text "fraction is an<<int fraction!"
//    happened at steps 748-749 or similar. But the OUTPUT TAIL shows the "<<" at position
//    750 in the preamble (the model wrote "<<" as its last preamble token).
//    
//    Actually, looking at preamble check:
//    "if next == eosToken { if |generated| > 0 && generated[|generated| - 1] == '<<' {...}}"
//    
//    THE ACTUAL SCENARIO: The model wrote "fraction is an<<" and then "int fraction!" in 
//    the PREAMBLE text. At some step during preamble, next = "<<". We DON'T intercept it.
//    We append generated + ["<<"]. Then next = "int". Append. Then "fraction". Append. Then "!".
//    At this point: preamble steps = 750. Phase 2 fires.
//    Phase 2: OpenConstrainedSpan → appends ANOTHER "<<" to generated.
//    Generated: [..., "<<", "int", "fraction", "!", "<<"]  ← two "<<" with only second being our span
//    Phase 3: generates into second "<<". 
//    Generated: [..., "<<", "int", "fraction", "!", "<<", "valid_expr", ">>"]
//    Evaluator picks LAST "<<" = our constrained span "<<valid_expr>>".
//    IF valid_expr is correct → success!
//    IF valid_expr is wrong → mode_H (wrong answer with constrained active)
//    BUT if Phase 3 generates ZERO tokens before budget → mode_G (unterminated).
//
//    OK so the mode_G "an<<int fraction!" tail seems to show that Phase 4 ALSO failed to close.
//    The last token in generated is "<<" → the second "<<" was appended by Phase 2.
//    Then Phase 3 tried to generate into it, but generated "int", "fraction", "!" somehow?
//    No, that can't be: those are from the preamble.
//
//    I think I'm misreading the output tail format. Let me try another interpretation:
//    The output tail is the LAST N characters of the generated string joined together.
//    The model's full output is: "...the result must be an integer...fraction is an<<int fraction!"
//    The "<<int fraction!" here is the MODEL's NATURAL output during the PREAMBLE (free text).
//    The model NEVER hit EOS and NEVER finished this sentence.
//    Preamble exits at 750 steps.
//    Phase 2: OpenConstrainedSpan → appends "<<" to generated.
//    Phase 3: generates some tokens.
//    Phase 4: CloseSpanWithinBudget.
//    
//    But the OUTPUT TAIL shows "...fraction is an<<int fraction!" and NOTHING AFTER IT.
//    This means: After the preamble, the generated content ENDED right there.
//    Phase 2 fired but added "<<" which doesn't show in the tail (maybe?)
//    OR: the whole strategy ENDED before Phase 2 could run.
//    
//    OH WAIT: maxSteps=900 and preambleLimit = maxSteps - 150 = 750.
//    In attempt 39, the preamble guard is "steps < preambleLimit = 750".
//    If the model wrote "fraction is an<<int fraction!" at steps 748-750:
//    - step 748: generates "<<", appended to generated
//    - step 749: generates "int", appended to generated  
//    - step 750: preambleLimit reached, loop exits (steps < 750 is false when steps=750)
//    
//    Wait: "steps := steps + 1" at start of loop iteration.
//    When steps=749, check: 749 < 750 → yes. UnconstrainedStep → steps=750.
//    Append "int" to generated. Check: 750 < 750 → false. Loop exits with steps=750.
//    
//    After preamble loop: steps=750, !insideConstrainedOut.
//    Phase 2: steps < maxSteps (750 < 900) → OpenConstrainedSpan → steps=751.
//    generated = [..., "<<", "int", "<<"]  ← wait no, "int" was appended during preamble
//    Hmm: [..., "<<", "int"] is what's in generated after preamble.
//    Phase 2 appends "<<": [..., "<<", "int", "<<"]
//    generated[|generated|-1] = "<<".
//    Phase 3a: 3 iterations of ConstrainedStep.
//    Should generate valid constrained tokens.
//    Phase 3b: CloseSpanIfComplete + ConstrainedStep loop.
//    Phase 4: CloseSpanWithinBudget.
//    Final: [..., "<<", "int", "<<", expr, ">>"]
//    Evaluator picks LAST "<<": the one at position |generated|-1 before Phase 3.
//    This span should have valid content and ">>".
//    
//    But mode_G says "zero_visible_spans" = NO complete "<<>>" found in output!
//    
//    The output tail "...fraction is an<<int fraction!" suggests the FINAL output doesn't 
//    have the "<<expr>>" pattern. Maybe: after Phase 2 appended "<<", Phase 3 generated
//    "int", " ", "fraction", "!" as constrained tokens (they ARE valid starts for formulas?).
//    Then Phase 4 ran out of budget.
//    
//    WAIT: If constrained prompt generates "int" → valid (int() starts formula).
//    Then "(": parenthesis opens. "fraction" → if this is a valid identifier. "!" → NOT VALID.
//    ConstrainedStep would NEVER generate "!" in a formula.
//    So the output "<<int fraction!" must be from the PREAMBLE phase (unconstrained).
//    The evaluator picks the LAST "<<" in generated.
//    The LAST "<<" in "...an<<int fraction!<<" would be OUR Phase 2 "<<".
//    Phase 3 generates into this: "<<valid>>" 
//    Evaluator finds "<<valid>>".
//    
//    UNLESS Phase 3/4 somehow FAILED and our "<<" is unclosed.
//    How? If Phase 3a's ConstrainedStep generates EOS → insideConstrainedOut=false.
//    Then Phase 3b and Phase 4 are skipped (insideConstrainedOut=false).
//    Our "<<" is left UNCLOSED.
//    
//    YES! This is the bug:
//    Phase 3a: ConstrainedStep → EOS → insideConstrainedOut=false, currentConstrainedOut=[].
//    Phase 3b: guard "insideConstrainedOut" → false → skipped.
//    Phase 4: guard "insideConstrainedOut" → false → skipped.
//    Our "<<" was appended (Phase 2) but never closed → mode_G!
//    
//    WHY DOES ConstrainedStep RETURN EOS HERE?
//    The constrained prompt = prompt + generated[..|generated|-0] = prompt + generated.
//    (since currentConstrainedOut = [], the prefix up to start of span is just generated.)
//    Wait: "constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|]"
//    When currentConstrainedOut = [], |currentConstrainedOut| = 0.
//    constrainedPrompt = prompt + generated[..|generated| - 0] = prompt + generated.
//    generated ends with "<<".
//    The model sees: "...fraction is an<<int fraction!<<" as its prompt for the constrained step.
//    The model is "inside" the last "<<" but the context before it contains "int fraction!<<"
//    which is garbage. The model might think EOS is appropriate here.
//    → ConstrainedStep returns EOS.
//    → insideConstrainedOut = false.
//    → "<<" left open → mode_G!
//
//    FIX: After Phase 3a generates EOS (while insideConstrainedOut was true), 
//    we should NOT leave the "<<" open. We need to either:
//    A. Run CloseSpanWithinBudget even when Phase 3a generated EOS
//    B. Close the span forcibly (append ">>" somehow)
//    C. Don't let Phase 3a set insideConstrainedOut=false on EOS - instead skip that token
//       and try again, OR immediately force-close.
//
//    The CORRECT fix: After Phase 3a loop, if insideConstrainedOut=false (EOS occurred)
//    but we were inside a span when Phase 3a started (we entered the span in Phase 2),
//    we need to handle the unclosed "<<".
//    
//    But this is complex to track. Simpler: Use CloseSpanWithinBudget REGARDLESS of
//    whether Phase 3a finished cleanly. Track whether we were inside a span at the START
//    of Phase 3a and whether we opened it in Phase 2.
//    
//    EVEN SIMPLER FIX:
//    In Phase 3a, when next==eosToken, DON'T set insideConstrainedOut=false.
//    Instead, BREAK out of the Phase 3a loop and let Phase 4 handle the closing.
//    Phase 4 uses CloseSpanWithinBudget which can handle an empty constrained span.
//    
//    But wait: if currentConstrainedOut=[] and insideConstrainedOut=true,
//    CloseSpanWithinBudget would generate tokens to complete the formula and close.
//    This is EXACTLY what we want!
//    
//    The postconditions say:
//    "!insideOut ==> currentOut == []"
//    "insideOut ==> parser.IsValidPrefix(currentOut)"
//    "|generatedOut| <= |generated| + budget"
//    
//    If we call CloseSpanWithinBudget with an empty currentConstrained and insideOut=true,
//    it will generate tokens and close the span. parser.IsValidPrefix([]) is true (precondition).
//    
//    FIX: In Phase 3a, when next==eosToken, DON'T modify insideConstrainedOut.
//    Instead, break out. Then Phase 3b won't run (since CloseSpanIfComplete would check).
//    Phase 4 runs with insideConstrainedOut=true and closes.
//
//    But the Dafny loop invariant requires we maintain consistency.
//    If we break without setting insideConstrainedOut=false on EOS:
//    - insideConstrainedOut remains true
//    - currentConstrainedOut remains valid (no change)
//    - Loop invariants maintained
//    - Phase 4 runs and closes cleanly
//
// OTHER UNTERMINATED SPANS:
// 2. "...Let\'s re-examine the input string carefully: "Each flowering plant costs {cur}{cost}".
//    \nIf the variable name<<cost is"
//    → Model wrote "name<<cost" as free text. Similar to above.
//    After preamble, Phase 2 opens "<<" → but Phase 3a generates EOS → unclosed.
//    FIX: Same as above.
//
// 3. The third unterminated span case is similar.
//
// MALFORMED CONTENT (mode_D: 1 example):
// "<<int(((sides * (sides - target)) - (sides // 2)**2) / (sides**2) * 100)>>"
// Contains "**" → evaluator rejects.
// This came from NATURAL output (used_constrained=no) → guidance failed to prevent **.
// We REMOVED the "do not use **" guidance in attempt 39.
// FIX: Re-add "Do not use ** for exponentiation" to guidance.
// For exponential problems: model should use pow(base, exp) or just (1+r)**d.
// Wait: but the evaluator ALSO rejects pow()? Let me think.
// Actually the correct answer for "n0 * (r+1)**d" is exactly that.
// The evaluator evaluates Python expressions. ** is valid Python.
// The evaluator's "contained **" error means it's in THEIR syntax check list,
// BUT if the answer is NUMERICALLY CORRECT, does it pass?
// From the failing rollout: the answer "int(((sides * (sides - target)) - (sides // 2)**2) / (sides**2) * 100)"
// → CORRECT ANSWER: "int((sides-target)/sides * 100) - 25"
// The formula is WRONG anyway! So even if ** were allowed, it would be wrong.
// Adding "do not use **" would make the model write a different (possibly correct) formula.
// → Add "Do not use **" guidance.
//
// STRATEGY FOR ATTEMPT 40:
// 1. Fix Phase 3a: Don't set insideConstrainedOut=false on EOS. Break instead.
//    Let Phase 4 handle closing. This fixes mode_G.
// 2. Re-add "Do not use ** for exponentiation" to guidance.
// 3. Keep everything else from attempt 36/39 (best structure).
// 4. Keep Phase 3b guard as "steps + 20 < maxSteps" (from attempt 39).
//
// ADDITIONAL IDEA: Since mode_H (15 examples) is the biggest failure (constrained active, wrong):
// The constrained phase is generating wrong formulas.
// One key issue: we're generating from AFTER preamble ends (mid-reasoning).
// The model's constrained generation quality improves if we give a better prefix.
//
// Can we improve the constrained prompt? Currently:
// constrainedPrompt = prompt + generated[..|generated| - |currentConstrainedOut|]
// This passes the FULL preamble text as context.
// The model sees all its reasoning + the "<<" and generates the formula.
// This should work IF the model has reasoned to the correct formula.
// The problem: model might not have finished reasoning before preamble ended.
//
// FOR MODE_H: The model didn't finish in 750 preamble steps.
// We force "<<" and generate constrained.
// The constrained content is WRONG because model is mid-reasoning.
//
// The ONLY fix: either let model finish (more preamble) or improve constrained generation quality.
//
// FROM THE FAILING ROLLOUT:
// "output tail: '...cost is {p4<<d (!'"  (from attempt 38 analysis)
// "output tail: '...{name}'<<relative_age>>'  (from attempt 36 analysis)
// These suggest the model is mid-sentence when we force "<<".
//
// IDEA: Use confidence-gated step instead of ConstrainedStep for Phase 3.
// ConfidenceGatedStep: uses LM's top token if it's valid, otherwise hard-masks.
// This might generate more "natural" formula tokens.
// But: we don't need soft confidence here. We want the model's reasoning to flow naturally.
// ConstrainedStep already gives pure LM preference (just hard-masks invalid tokens).
//
// IDEA: After Phase 2 opens "<<", do several ConstrainedSymbol calls instead of ConstrainedStep.
// ConstrainedSymbol generates up to maxSymbolTokens and takes the longest valid prefix.
// This might give better multi-token chunks.
// But: ConstrainedSymbol with stepTokenBudget=1 is just ConstrainedStep.
// With larger stepTokenBudget: might generate better chunks.
// Available: stepTokenBudget is a parameter we're given (=1 in this task).
// So ConstrainedSymbol(stepTokenBudget=1) = ConstrainedStep.
//
// STAYING WITH CURRENT APPROACH:
// The main fix is the mode_G bug (EOS in Phase 3a leaving "<<" unclosed).
// Secondary fix: "Do not use **" guidance.
// These should recover from 36.7% back toward 40.8% (attempt 36).
//
// ADDITIONAL: The "steps + 20 < maxSteps" in Phase 3b guard might be too conservative.
// With preamble at 750, we have 150 steps for phases 2-4.
// Phase 2: 1 step. Phase 3a: up to 3 steps. Phase 3b: up to (150 - 1 - 3 - 20) = 126 steps.
// Phase 4: at least 20 steps.
// 126 iterations in Phase 3b, each taking 2 steps = 63 steps MAX in Phase 3b.
// 63 constrained tokens should be enough for any formula.
// This seems fine. Keep it.
//
// Wait: "steps + 20 < maxSteps" = 750 + 1 (phase 2) + 3 (phase 3a) + x < 900
// = 754 + x < 900 → x < 146.
// Phase 3b runs while steps ≤ 879. Each iteration: +2 steps.
// At step 879: loop exits. Phase 4 gets 900-879 = 21 ≥ 20 steps. Good.
//
// FINAL STRATEGY SUMMARY:
// All from attempt 36 (best baseline) with targeted fixes:
// 1. Phase 3a EOS fix: break instead of setting insideConstrainedOut=false
// 2. Guidance: re-add "Do not use **"
// 3. Phase 3b guard: "steps + 20 < maxSteps" (from attempt 39, helps with mode_G)
//
// CSD_RATIONALE_END

// CSD_PROOF_SKETCH_BEGIN
//
// 1. parser_validity: insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
//
//    Initialization: matches precondition exactly.
//
//    Phase 1 preamble loop:
//      insideConstrainedOut is never modified (stays false = value from init).
//      The invariant "!insideConstrainedOut" is a loop invariant.
//      EOS break: either early return (postconditions hold by precondition + invariant) or
//        hitEosWithOpenSpan=true, break. insideConstrainedOut still false → implication vacuous. ✓
//      Non-EOS: generated grows, insideConstrainedOut unchanged → vacuous. ✓
//
//    Phase 2 (span entry):
//      EnterObservedConstrainedSpan: insideConstrainedOut=true, currentConstrainedOut=[].
//        parser.IsValidPrefix([]) holds by precondition. ✓
//      OpenConstrainedSpan: same. ✓
//
//    Phase 3a (min token loop):
//      EOS: break WITHOUT modifying insideConstrainedOut/currentConstrainedOut.
//        Invariant maintained (was valid before, unchanged). ✓
//      Non-EOS: AppendConstrainedToken postcondition guarantees
//        parser.IsValidPrefix(currentConstrainedOut + [next]). ✓
//
//    Phase 3b (close loop):
//      CloseSpanIfComplete: closes (insideConstrainedOut=false → vacuous) or no-op. ✓
//      ConstrainedStep + AppendConstrainedToken: same validity guarantee as Phase 3a. ✓
//      EOS in ConstrainedStep: break. insideConstrainedOut/currentConstrainedOut unchanged.
//        Invariant maintained. ✓ (We do set insideConstrainedOut=false here since we're in
//        the main loop, not Phase 3a special handling.)
//
//    Phase 4 (CloseSpanWithinBudget):
//      Postcondition: insideOut ==> parser.IsValidPrefix(currentOut). ✓
//
// 2. progress: |generated| <= |generatedPrefix| + steps
//
//    Initialization: equality holds. ✓
//
//    Phase 1: Each iteration: +1 step, +0 or +1 token. Bound maintained. ✓
//
//    Phase 2:
//      EnterObservedConstrainedSpan: +0 steps, no append. Bound unchanged. ✓
//      OpenConstrainedSpan: +1 step, +1 token ("<<"). Both sides +1. ✓
//
//    Phase 3a:
//      EOS (break): +1 step, +0 tokens. Bound maintained. ✓
//      Non-EOS: +1 step, +1 token. Both sides +1. ✓
//      Termination: preCloseSteps is bounded by minTokensToGenerate. ✓
//
//    Phase 3b (guard: steps + 20 < maxSteps):
//      CloseSpanIfComplete: +1 step, +0 or +1 token (">>" when closing).
//        Closed: steps ≤ maxSteps, |generated| ≤ |generatedPrefix| + steps. ✓
//      ConstrainedStep: +1 step, +0 or +1 token.
//      Guard ensures at least 20 steps remain after each iteration.
//      |generated| ≤ |generatedPrefix| + steps ≤ |generatedPrefix| + maxSteps. ✓
//      Decreases: maxSteps - steps, strictly increasing each iteration. ✓
//
//    Phase 4:
//      closeBudget = maxSteps - steps.
//      CloseSpanWithinBudget: |generatedOut| ≤ |generated| + closeBudget = |generatedPrefix| + maxSteps.
//      steps := maxSteps. ✓
//
// 3. cost <= maxSteps: cost := steps; steps ≤ maxSteps throughout. ✓
//
// 4. Progress postcondition: maxSteps == 0 || cost > 0 || ...
//    maxSteps > 0 → preambleLimit ≥ 0. If preambleLimit ≥ 1: Phase 1 runs ≥1 step → cost ≥ 1. ✓
//    If preambleLimit = 0 and !insideConstrained: Phase 2 runs (OpenConstrainedSpan, +1 step). ✓
//    If insideConstrained=true: Phase 3a runs, ConstrainedStep → steps ≥ 1 → cost ≥ 1. ✓
//
// CSD_PROOF_SKETCH_END

generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

// Guidance: instruct model to use exact variable names, int(), //, include all variables,
// and explicitly forbid ** (to prevent malformed syntax) and {braces}.
var guidance: string := "Solve the problem step by step. Write your final answer as a single symbolic expression inside << >>. Use EXACT variable names from the problem (the words inside {} braces, WITHOUT the braces themselves - e.g., write n1 not {n1}). NEVER include { or } characters inside << >>. Use // for integer division when converting time (minutes to hours). Use int() to wrap the entire expression when the result must be an integer count or monetary amount. Make sure to include ALL variables from the problem in your formula - do not omit any variable. Do not use ** for exponentiation.";
helpers.AppendTaskGuidance(lm, guidance);

var steps: nat := 0;
var hitEosWithOpenSpan: bool := false;

// Phase 1: PURE unconstrained preamble.
// Early-return on EOS UNLESS the last token is "<<" (trailing open span).
// Reserve 150 steps for constrained generation phase.
var preambleLimit: nat := if maxSteps > 150 then maxSteps - 150 else 0;

if !insideConstrainedOut {
  while steps < preambleLimit
    invariant 0 <= steps <= maxSteps
    invariant lm.ValidTokensIdsLogits()
    invariant !insideConstrainedOut ==> currentConstrainedOut == []
    invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
    invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|
    invariant |generated| <= |generatedPrefix| + steps
    invariant !insideConstrainedOut
    decreases preambleLimit - steps
  {
    var next := helpers.UnconstrainedStep(lm, prompt, generated);
    steps := steps + 1;
    if next == eosToken {
      // Check if model left a trailing "<<" - if so, we need to handle it in Phase 2
      if |generated| > 0 && generated[|generated| - 1] == "<<" {
        hitEosWithOpenSpan := true;
        break;
      } else {
        // Normal EOS: trust the model's natural output
        cost := steps;
        return;
      }
    } else {
      generated := generated + [next];
      // Do NOT intercept "<<" during preamble - pure unconstrained
    }
  }
}

// Phase 2: Enter or open the constrained span
if !insideConstrainedOut && steps < maxSteps {
  if hitEosWithOpenSpan && |generated| > 0 && generated[|generated| - 1] == "<<" {
    // Model already opened a span (trailing "<<"), enter it without adding another delimiter
    var og, oi, oc := helpers.EnterObservedConstrainedSpan(lm, generated);
    generated := og;
    insideConstrainedOut := oi;
    currentConstrainedOut := oc;
    // EnterObservedConstrainedSpan costs 0 steps
  } else {
    // Normal case: force open a new constrained span for the final answer
    var og, oi, oc := helpers.OpenConstrainedSpan(lm, generated);
    generated := og;
    insideConstrainedOut := oi;
    currentConstrainedOut := oc;
    steps := steps + 1;
  }
}

// Phase 3a: Generate minimum constrained tokens BEFORE attempting to close.
// KEY FIX: When ConstrainedStep returns EOS, BREAK without setting insideConstrainedOut=false.
// This allows Phase 4 (CloseSpanWithinBudget) to handle the open span cleanly.
// Previously, setting insideConstrainedOut=false on EOS left the "<<" unclosed (mode_G bug).
var minTokensToGenerate: nat := 3;
var preCloseSteps: nat := 0;

while preCloseSteps < minTokensToGenerate && steps < maxSteps && insideConstrainedOut
  invariant 0 <= steps <= maxSteps
  invariant lm.ValidTokensIdsLogits()
  invariant !insideConstrainedOut ==> currentConstrainedOut == []
  invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
  invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|
  invariant |generated| <= |generatedPrefix| + steps
  decreases minTokensToGenerate - preCloseSteps
{
  var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];
  var next := helpers.ConstrainedStep(lm, parser, constrainedPrompt, currentConstrainedOut, eosToken);
  steps := steps + 1;
  preCloseSteps := preCloseSteps + 1;
  if next == eosToken {
    // KEY FIX: Don't close constrained state on EOS here.
    // Break out and let Phase 4 handle the open span with CloseSpanWithinBudget.
    break;
  } else {
    var ag, ai, ac := helpers.AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, next);
    generated := ag;
    insideConstrainedOut := ai;
    currentConstrainedOut := ac;
  }
}

// Phase 3b: Continue constrained generation with CloseSpanIfComplete check each step.
// Reserve 20 steps for Phase 4 to guarantee CloseSpanWithinBudget can close the span.
while steps + 20 < maxSteps && insideConstrainedOut
  invariant 0 <= steps <= maxSteps
  invariant lm.ValidTokensIdsLogits()
  invariant !insideConstrainedOut ==> currentConstrainedOut == []
  invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
  invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|
  invariant |generated| <= |generatedPrefix| + steps
  decreases maxSteps - steps
{
  // Check if we can close the span now
  var cg, ci, cc, closed := helpers.CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut);
  steps := steps + 1;
  if closed {
    generated := cg;
    insideConstrainedOut := ci;
    currentConstrainedOut := cc;
  } else {
    // Generate the next constrained token using pure LM preference
    if steps < maxSteps {
      var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];
      var next := helpers.ConstrainedStep(lm, parser, constrainedPrompt, currentConstrainedOut, eosToken);
      steps := steps + 1;
      if next == eosToken {
        insideConstrainedOut := false;
        currentConstrainedOut := [];
      } else {
        var ag, ai, ac := helpers.AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, next);
        generated := ag;
        insideConstrainedOut := ai;
        currentConstrainedOut := ac;
      }
    }
  }
}

// Phase 4: If still inside span and budget allows, use CloseSpanWithinBudget
// to reach a completable state and emit ">>".
// With guard "steps + 20 < maxSteps" in Phase 3b, we have at least 20 steps here.
// This also handles the case where Phase 3a broke on EOS (leaving an open span).
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
