// CSD_RATIONALE_BEGIN
// Root cause analysis: "0 examples evaluated" means verification/compilation failure.
// The previous attempt had complex nested CloseSpanWithinBudget calls that caused
// verification issues due to step accounting problems.
//
// The fundamental issue: the task is gsm_symbolic (math word problems) and needs
// to produce numeric answers inside << >> delimiters. The evaluator checks for
// << >> syntax and accuracy.
//
// Strategy: Use the proven GenerateWithManagedSpan helper which handles all the
// free/constrained/close state machine internally with verified postconditions.
// This avoids all the complex manual span management that was causing verification failures.
//
// GenerateWithManagedSpan:
// - Handles free text until "<<" appears
// - Switches to constrained decoding inside the span
// - Closes the span with ">>" when parser accepts completion
// - Returns valid state with all invariants satisfied
// - Cost: at most maxSteps
//
// Add task guidance to hint the model to produce numeric answers in << >> spans.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity:
//   GenerateWithManagedSpan's postcondition guarantees:
//   insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
//   This is directly satisfied by the helper's contract.
//
// progress:
//   GenerateWithManagedSpan's postcondition guarantees:
//   |generated| <= |generatedPrefix| + maxSteps
//   cost <= maxSteps (set to return value which is at most maxSteps)
//   The helper internally manages all step counting and output growth bounds.
// CSD_PROOF_SKETCH_END
generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

var guidance: string := "Solve the math problem. Show your work. Place your final numeric answer inside << >> delimiters, for example: <<42>>.";
helpers.AppendTaskGuidance(lm, guidance);

var genOut, insideOut, currentOut := helpers.GenerateWithManagedSpan(
  lm, parser, prompt, generatedPrefix, insideConstrained, currentConstrained,
  maxSteps, validTokenGroups, 4.0, 12, eosToken
);

generated := genOut;
insideConstrainedOut := insideOut;
currentConstrainedOut := currentOut;
cost := maxSteps;
