// CSD_RATIONALE_BEGIN
// Analysis of failure patterns from prior attempts:
// 
// Best attempt (58.3%): Opens visible << span, then uses CloseSpanWithinBudget.
// But metrics show: "Examples with visible <<: 0/300", "Examples with unmatched visible close >>: 300/300"
// This means << is NOT appearing visibly even though we call OpenConstrainedSpan.
// The >> IS appearing (from CloseSpanWithinBudget), but without visible <<.
// Answer source: "hidden_or_task_extractor" for correct answers in attempt 18,
// but "text_fallback" for all 133 wrong in attempt 21.
//
// Wait - re-reading attempt 21's metrics:
// "Examples with visible <<: 0/300" - NONE have visible <<
// "Examples with complete visible spans: 0/300"
// "Examples with unmatched visible close: 300/300" - ALL have >> without <<
// "Answer extraction source: hidden_or_task_extractor 0/300, text_fallback? wait..."
// Actually: "answer_source: text_fallback (100%)" for wrong examples (mode_C).
//
// The key observation: in attempt 21 (55.7%), all 300 examples had unmatched >>
// but no visible <<. The 167 correct ones used constrained activity.
// "answer_source: hidden_or_task_extractor" for correct ones from attempt 18.
//
// The actual issue: CloseSpanWithinBudget emits >> but the evaluator extracts from
// the constrained (hidden) span, not from visible << >>. The best attempt (58.3%)
// had correct answers because the hidden constrained span was being extracted.
//
// The failure is purely semantic: the model generates wrong SQL.
// 
// Looking at the three rollout examples:
// 1. "SELECT COUNT(*) FROM pets" - missing DISTINCT pettype
//    - Correct: "select count ( distinct pettype ) from pets"
//    - Model ignored the "distinct type" in the question
// 2. Uses JOIN when simple WHERE would work (Anguilla -> continent)
//    - The guidance "Do NOT use JOIN unless required" failed
// 3. Adds extra columns (MAX zip_code, etc.) when not asked
//
// The root cause is the LM generates overly complex SQL despite guidance.
// The guidance isn't being followed well enough.
//
// New approach: Instead of just CloseSpanWithinBudget, use a step-by-step
// constrained generation loop with RepetitionPenaltyStep to avoid repetition.
// More importantly, try different guidance that more specifically addresses failures.
//
// Key insight from rollout 2: "SELECT T2.continent FROM country AS T1 JOIN city..."
// The correct answer is "select continent from country where name = 'Anguilla'"
// The model is using JOIN country+city when the answer is just country WHERE.
//
// The guidance change that might help:
// - "Look carefully at the FROM clause - only include tables that have columns needed"
// - "For filtering by name/id, use WHERE not JOIN"  
// - "Do NOT select extra columns not requested"
//
// More importantly: use RepetitionPenaltyStep inside the constrained loop to
// prevent the repetition problem (rollout 3 with very long output).
//
// Actually, the avg tokens is 26.19 with max 111 - not that many. The CloseSpanWithinBudget
// is limiting generation to just the SQL query.
//
// Better strategy: Let the model generate MORE freely first (up to 20 tokens of "thinking")
// before entering the constrained span. This might help the model produce better SQL.
// 
// BUT - previous attempt 19 tried more tokens with worse results.
//
// Alternative: Try the best approach (18) but with DIFFERENT guidance targeted
// at the specific patterns of failure:
// - Anguilla case: single-table query, just WHERE
// - pets DISTINCT: model missed the DISTINCT requirement
// - owner dogs: overly complex with extra MAX columns
//
// The most impactful fix would be guidance that:
// 1. Forces the model to re-read the question carefully for DISTINCT
// 2. Prevents adding extra columns beyond what's requested
// 3. Prevents unnecessary JOINs
//
// Let me try a more focused, cleaner guidance approach combined with the
// working structure from attempt 18.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity: insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
//   Phase 1 (unconstrained prefix loop): insideConstrainedOut is only set to true
//   when RenderedEndsWith([next], "<<"), at which point currentConstrainedOut := []
//   which satisfies parser.IsValidPrefix([]) by precondition.
//   Phase 2 (OpenConstrainedSpan): postcondition ensures insideConstrainedOut=true
//   and currentConstrainedOut=[], with parser.IsValidPrefix([]) by precondition.
//   Phase 3 (CloseSpanWithinBudget): postcondition guarantees
//   !insideConstrainedOut ==> currentConstrainedOut == [] and
//   insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut).
//
// progress: |generated| <= |generatedPrefix| + steps
//   Phase 1: each iteration appends at most 1 token to generated and increments
//   steps by 1, preserving the invariant.
//   Phase 2: OpenConstrainedSpan appends exactly 1 token ("<<"), steps += 1.
//   Phase 3: CloseSpanWithinBudget postcondition |generatedOut| <= |generated| + remainingBudget,
//   and steps += remainingBudget <= maxSteps. Total |generated| growth <= maxSteps.
//   cost := steps <= maxSteps satisfies ensures cost <= maxSteps.
//   When maxSteps > 0: phase 1 always takes >= 1 step (or phase 2 does), so cost > 0.
// CSD_PROOF_SKETCH_END
generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

if maxSteps == 0 {
  return;
}

// Guidance targeting specific failure modes observed in rollouts:
// 1. Missing DISTINCT when question says "distinct type of"
// 2. Using JOIN+alias when simple single-table WHERE suffices
// 3. Adding extra unrequested columns (MAX, etc.)
var guidance: string := "Write the minimal correct SQL query. Read the question carefully: if it says 'distinct' or 'different' use DISTINCT. Only SELECT the exact columns asked for - no extra columns. Only JOIN tables if you absolutely need columns from both. Use simple WHERE for filtering. Do not use table aliases.";
helpers.AppendTaskGuidance(lm, guidance);

var steps: nat := 0;

// Phase 1: Generate free prefix (allows "SQL: " to be emitted naturally)
// Use a small budget of 5 tokens
var prefixBudget: nat := 5;
if prefixBudget > maxSteps {
  prefixBudget := maxSteps;
}

while steps < prefixBudget && !insideConstrainedOut
  invariant 0 <= steps <= maxSteps
  invariant lm.ValidTokensIdsLogits()
  invariant !insideConstrainedOut ==> currentConstrainedOut == []
  invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
  invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|
  invariant |generated| <= |generatedPrefix| + steps
  decreases prefixBudget - steps
{
  var next := helpers.UnconstrainedStep(lm, prompt, generated);
  steps := steps + 1;
  if next == eosToken {
    break;
  }
  generated := generated + [next];
  if RenderedEndsWith([next], "<<") {
    insideConstrainedOut := true;
    currentConstrainedOut := [];
  }
}

// Phase 2: Explicitly open the constrained span with visible "<<"
if !insideConstrainedOut && steps < maxSteps {
  var og, oi, oc := helpers.OpenConstrainedSpan(lm, generated);
  generated := og;
  insideConstrainedOut := oi;
  currentConstrainedOut := oc;
  steps := steps + 1;
}

// Phase 3: Generate constrained SQL content and close the span
// Use CloseSpanWithinBudget with all remaining budget
if insideConstrainedOut && steps < maxSteps {
  var remainingBudget: nat := maxSteps - steps;
  var cg, ci, cc := helpers.CloseSpanWithinBudget(
    lm, parser, prompt, generated, currentConstrainedOut, eosToken, remainingBudget
  );
  generated := cg;
  insideConstrainedOut := ci;
  currentConstrainedOut := cc;
  steps := steps + remainingBudget;
}

cost := steps;
