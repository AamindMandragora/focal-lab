    // CSD_RATIONALE_BEGIN
// Current best: 39-40% accuracy, 98% syntax. Gap: 1pp accuracy.
// Primary failure: 59/100 syntax_valid_semantic_mismatch.
// The model generates valid SQL but wrong semantics.
//
// Key observations from failing examples:
// 1. "List cartoons alphabetically" -> model omits ORDER BY
// 2. "cars with larger accelerate than car with largest horsepower" -> model uses MAX(horsepower) instead of subquery with ORDER BY/LIMIT
// 3. "semester when both Master and Bachelor enrolled" -> model uses OR instead of INTERSECT
// 4. "citizenship shared by singers before 1945 and after 1955" -> model generates INTERSECT INTERSECT! (malformed)
//
// The model's own preferences (ConfidenceGatedStep) lead to semantically wrong queries.
// The model is a small 1.5B model that tends toward simpler patterns.
//
// Strategy change: try AdaptiveConstrainedStep instead of ConfidenceGatedStep.
// AdaptiveConstrainedStep applies hard masking always, with group boosts when narrow.
// This is stronger constraint than ConfidenceGatedStep but may push model toward
// different (possibly more correct) token choices.
//
// Actually, the issue is not the helper but the guidance. The model generates
// wrong SQL because it doesn't understand the query semantics.
//
// Let me think about what guidance could help:
// - Current best: "Write a concise, correct SQL query. Use only the tables and columns
//   from the schema. Output the query directly without extra conditions."
// - This is already quite good at 39-40%.
//
// The specific failure cases suggest:
// 1. Model doesn't add ORDER BY when needed
// 2. Model uses MAX() instead of subquery + ORDER BY LIMIT 1
// 3. Model uses OR instead of INTERSECT
//
// These are semantic issues that guidance might not fix easily.
//
// Alternative approach: try AdaptiveConstrainedStep with validTokenGroups as boost.
// The validTokenGroups might contain SQL keywords that could bias the model.
//
// Actually, let me try a different guidance that emphasizes reading the question carefully
// and matching the exact semantics. The key insight is that the model needs to
// understand "both X and Y" means INTERSECT, "largest" means subquery or ORDER BY.
//
// New guidance attempt: "Generate SQL that exactly answers the question. Use INTERSECT
// for 'both', ORDER BY for 'alphabetical/largest/smallest', subqueries when comparing
// to extremes. Match the question semantics precisely."
//
// Wait - attempt 9 tried adding specific clause hints and dropped to 33%.
// The model got confused by the extra instructions.
//
// The safest approach: keep the exact same structure but try a tiny guidance variation.
// The 1pp gap might be bridged by a slightly different guidance phrasing.
//
// Let me try: "Write the correct SQL query for the question. Use the exact schema tables
// and columns. Be precise about the SQL semantics."
//
// Actually, looking at the data more carefully:
// - 39/100 correct with current approach
// - The 59 wrong ones are syntactically valid but semantically wrong
// - The model is generating plausible but wrong SQL
//
// The constraint approach can't fix semantic understanding. The only lever we have
// is guidance text.
//
// Let me try a guidance that explicitly mentions common patterns:
// "Write a correct SQL query. For 'both A and B', use INTERSECT. For 'alphabetical order',
// add ORDER BY. Use subqueries when comparing to max/min values."
//
// But attempt 9 showed that adding specific clause hints hurts performance.
//
// The best approach is to keep the exact same strategy that got 39-40% and try
// a slightly different guidance. The variance between attempts at the same strategy
// seems to be about 1pp, so we might just get lucky.
//
// Final decision: use exact same strategy as best attempt (ConfidenceGatedStep)
// but try a guidance that is slightly more specific about semantic correctness
// without being too prescriptive.
//
// New guidance: "Write a concise, correct SQL query answering the question exactly.
// Use only schema tables and columns. Match the precise semantics of the question."
//
// This is slightly more focused on "answering exactly" and "precise semantics"
// without listing specific SQL clauses that confused the model in attempt 9.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity:
//   1. OpenConstrainedSpan: sets insideConstrainedOut := true, currentConstrainedOut := [].
//      parser.IsValidPrefix([]) holds by precondition. Invariant established.
//   2. ConfidenceGatedStep returns next where either next == eosToken (no append, invariant unchanged)
//      or parser.IsValidPrefix(currentConstrainedOut + [next]) holds (by helper contract).
//      AppendConstrainedToken sets new currentConstrainedOut = old + [next], preserving validity.
//   3. CloseConstrainedSpan: sets insideConstrainedOut := false, implication vacuously true.
//   4. EOS break: no state change, invariant holds.
//
// progress:
//   OpenConstrainedSpan: steps += 1, |generated| += 1 ("<<"). Invariant: |gen| = |genPrefix| + 1 = |genPrefix| + steps. ✓
//   ConfidenceGatedStep + AppendConstrainedToken: steps += 1, |generated| += 1. ✓
//   ConfidenceGatedStep returning EOS: steps += 1, |generated| unchanged <= |genPrefix| + (steps-1) < |genPrefix| + steps. ✓
//   CloseConstrainedSpan: steps += 1, |generated| += 1 (">>"). ✓
//   All branches: |generated| <= |generatedPrefix| + steps preserved.
// CSD_PROOF_SKETCH_END

generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

var steps: nat := 0;

// Guidance: slightly refined from best attempt to emphasize precise semantics
helpers.AppendTaskGuidance(lm, "Write a concise, correct SQL query answering the question exactly. Use only schema tables and columns. Match the precise semantics of the question.");

// Phase 1: Force open the constrained span (costs 1 step)
if steps < maxSteps && !insideConstrainedOut {
  var openGenerated, openInside, openCurrent := helpers.OpenConstrainedSpan(lm, generated);
  generated := openGenerated;
  insideConstrainedOut := openInside;
  currentConstrainedOut := openCurrent;
  steps := steps + 1;
}

// Phase 2: Generate constrained SQL inside the span using ConfidenceGatedStep
// ConfidenceGatedStep uses the model's preferred token when parser-valid,
// only hard-masking when the model's choice would violate the parser.
while steps < maxSteps && insideConstrainedOut
  invariant 0 <= steps <= maxSteps
  invariant lm.ValidTokensIdsLogits()
  invariant !insideConstrainedOut ==> currentConstrainedOut == []
  invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
  invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|
  invariant |generated| <= |generatedPrefix| + steps
  decreases maxSteps - steps
{
  if parser.IsCompletePrefix(currentConstrainedOut) {
    var closedGenerated, closedInside, closedCurrent := helpers.CloseConstrainedSpan(
      lm, parser, generated, currentConstrainedOut
    );
    generated := closedGenerated;
    insideConstrainedOut := closedInside;
    currentConstrainedOut := closedCurrent;
    steps := steps + 1;
  } else {
    var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];
    var next, wasConstrained := helpers.ConfidenceGatedStep(
      lm, parser, constrainedPrompt, currentConstrainedOut, eosToken
    );
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
  }
}

cost := steps;
