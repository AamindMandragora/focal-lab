// CSD_RATIONALE_BEGIN
// Analysis of failure modes:
// - Best attempt (58.3%) generates correct SQL syntax but semantically wrong content
// - The model is generating SQL with wrong columns/tables/conditions
// - Key failures: over-selecting columns, wrong aggregations, wrong JOINs
// - "token_budget_band: low" for 238/244 wrong examples - using very few tokens
// - avg 50.19 tokens/example with median 31.5 - queries are too short/truncated
//
// The issue is the model generates wrong SQL. Looking at the rollouts:
// 1. "SELECT t.age, t.hometown FROM teacher t" instead of "select age, hometown from teacher"
//    - Wrong: uses table alias, correct should be simple
// 2. "SELECT COUNT(DISTINCT airlines.uid) AS num_airlines FROM airlines JOIN flights..."
//    - Wrong: adds unnecessary JOIN, correct is simple WHERE clause
// 3. Long repetitive query - model enters repetition loop
//
// The fundamental problem: The model is generating SQL with:
// a) Unnecessary table aliases (t.age instead of age)
// b) Unnecessary JOINs
// c) Wrong column selections
// d) Repetition loops for complex queries
//
// The 58.3% accuracy with 100% syntax means the SQL is syntactically valid but
// semantically wrong. The model is capable but misled.
//
// Root cause hypothesis:
// The guidance "Select ONLY the columns asked for" is being followed literally but
// the model still adds unnecessary qualifiers, JOINs, etc.
//
// Key structural fix: Use RepetitionPenaltyStep inside the constrained loop instead
// of CloseSpanWithinBudget to prevent repetition loops. But RepetitionPenaltyStep
// was not in the best attempt. Also try stronger guidance.
//
// More importantly: the failure mode is "valid_nonfinal_spans_only 300/300" which means
// the spans are being treated as non-final. This means the answer extraction is
// looking at the generated text but finding the SQL before the ">>" as a non-final span.
//
// Wait - looking at best attempt again:
// - 58.3% accuracy, 300/300 valid spans
// - answer_source: hidden_or_task_extractor for successful ones
// - The SQL is being extracted and evaluated
//
// The question is: why do 244 fail? The SQL is syntactically valid but semantically wrong.
//
// Strategy to improve:
// 1. Better guidance that emphasizes simple queries without aliases unless necessary
// 2. Use a step-by-step constrained generation loop with RepetitionPenaltyStep
//    to avoid the long repetitive query problem
// 3. Keep the visible << >> structure that worked in attempt 18
//
// Specific guidance changes:
// - "Do NOT use table aliases or AS unless the question requires it"
// - "Keep the query as simple as possible"
// - "Use COUNT(*) for counting, not COUNT(DISTINCT col)"
// - "Do NOT add JOINs unless required by the question"
//
// Implementation:
// - Use CloseSpanWithinBudget with improved guidance (same as best attempt)
// - Add stronger/more specific guidance
// - Keep the same structure that achieved 58.3%
//
// The guidance is the key variable to change since the structure works.
// Let me craft guidance that addresses the specific failure patterns observed.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity: insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
//   Phase 1 (unconstrained loop): we never set insideConstrainedOut=true unless
//   RenderedEndsWith([next], "<<") is true, setting currentConstrainedOut := [] which
//   satisfies parser.IsValidPrefix([]) by the method precondition. No other modification.
//   Phase 2 (OpenConstrainedSpan): postcondition guarantees insideConstrainedOut=true
//   and currentConstrainedOut=[], with parser.IsValidPrefix([]) by precondition.
//   Phase 3 (CloseSpanWithinBudget): postcondition guarantees
//   !insideConstrainedOut ==> currentConstrainedOut == [] and
//   insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut).
//
// progress: |generated| <= |generatedPrefix| + steps
//   Phase 1: each loop iteration appends at most 1 token to generated and increments
//   steps by 1, preserving the invariant.
//   Phase 2: OpenConstrainedSpan appends exactly 1 token ("<<"), steps incremented by 1.
//   Phase 3: CloseSpanWithinBudget postcondition gives
//   |generatedOut| <= |generated| + remainingBudget, and steps := steps + remainingBudget
//   = maxSteps. So |generated| <= |generatedPrefix| + maxSteps.
//   cost := steps <= maxSteps satisfies ensures cost <= maxSteps.
//   When maxSteps > 0: phase 1 takes >= 1 step or phase 2 takes 1 step, so cost > 0.
// CSD_PROOF_SKETCH_END
generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

if maxSteps == 0 {
  return;
}

// Guidance targeting the specific failure modes observed:
// - Over-complex queries with unnecessary JOINs
// - Wrong aggregation functions  
// - Column aliases and table qualifiers
// - Repetition loops
var guidance: string := "Write the simplest correct SQL. Use COUNT(*) for counting rows. Do NOT use table aliases. Do NOT add JOIN unless the question explicitly requires data from multiple tables. Do NOT use DISTINCT unless explicitly asked. Select only the exact columns mentioned in the question. Match the exact column and table names from the schema.";
helpers.AppendTaskGuidance(lm, guidance);

var steps: nat := 0;

// Phase 1: Generate free prefix up to 6 steps (allows "SQL", ":", " " tokens)
var prefixBudget: nat := 6;
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

// Phase 2: Explicitly open constrained span with visible "<<"
if !insideConstrainedOut && steps < maxSteps {
  var og, oi, oc := helpers.OpenConstrainedSpan(lm, generated);
  generated := og;
  insideConstrainedOut := oi;
  currentConstrainedOut := oc;
  steps := steps + 1;
}

// Phase 3: Generate constrained SQL and close span
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
