// CSD_RATIONALE_BEGIN
// Strategy: Use GenerateWithManagedSpan to handle the full decode loop for SQL generation.
// The spider dataset requires generating SQL queries, which are structured outputs.
// We use AdaptiveConstrainedStep inside constrained spans to ensure parser validity,
// with group boosts from validTokenGroups to prefer schema-relevant tokens.
// The managed span helper handles delimiter emission, state transitions, and proof obligations.
// We append task guidance to help the model understand it should generate SQL.
// CSD_RATIONALE_END

// CSD_PROOF_SKETCH_BEGIN
// Invariant 1 (parser_validity): insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
//   - GenerateWithManagedSpan maintains this invariant internally on every path:
//     outside a span, currentConstrainedOut == [] (trivially valid by parser.IsValidPrefix([]) precondition);
//     inside a span, AdaptiveConstrainedStep only appends parser-valid tokens, so currentConstrainedOut
//     remains a valid prefix; CloseConstrainedSpan resets currentConstrainedOut to [] and sets
//     insideConstrainedOut to false, preserving the invariant.
//
// Invariant 2 (progress): |generated| <= |generatedPrefix| + steps
//   - GenerateWithManagedSpan consumes at most maxSteps token-steps total.
//     Each step appends at most one visible token to generated (delimiter tokens count as one step
//     and add at most one token to visible output). EOS terminates without appending, but still
//     consumes one step. Therefore after k steps, |generated| <= |generatedPrefix| + k <= |generatedPrefix| + maxSteps.
// CSD_PROOF_SKETCH_END

{
  // Initialize all out-parameters
  generated := generatedPrefix;
  insideConstrainedOut := insideConstrained;
  currentConstrainedOut := currentConstrained;
  cost := 0;

  if maxSteps == 0 {
    return;
  }

  // Append task guidance to help the model generate valid SQL
  helpers.AppendTaskGuidance(lm, "Generate a valid SQL query answering the question. Output the SQL directly.");

  // Use GenerateWithManagedSpan to handle the full decode loop
  // This handles free preamble, constrained span entry/exit, and proof obligations
  var boostAmount := 4.0;
  var narrowThreshold := 12;

  generated, insideConstrainedOut, currentConstrainedOut := helpers.GenerateWithManagedSpan(
    lm, parser, prompt, generatedPrefix,
    insideConstrained, currentConstrained,
    maxSteps,
    validTokenGroups,
    boostAmount,
    narrowThreshold,
    eosToken
  );

  cost := maxSteps;
}
