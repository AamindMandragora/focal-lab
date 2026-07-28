// CSD_RATIONALE_BEGIN
// SMILES isocyanate CSD. The task requires a single valid SMILES string for an
// isocyanate molecule (containing the -N=C=O functional group). The strategy:
// 1. Appends task guidance to steer the LM toward isocyanate SMILES.
// 2. Opens a constrained span immediately (no free preamble needed since the
//    answer contract is just a SMILES string).
// 3. Inside the span, uses AdaptiveConstrainedStep with repetition penalty to
//    generate diverse, valid SMILES tokens under parser control.
// 4. Closes the span when the parser reports a complete SMILES parse.
// State tracked: steps (budget), insideConstrainedOut, currentConstrainedOut.
// The parser enforces SMILES grammar validity at every token.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity:
//   OpenConstrainedSpan sets insideConstrainedOut := true and currentConstrainedOut := [],
//   which satisfies parser.IsValidPrefix([]) by precondition. In the constrained
//   generation loop, AdaptiveConstrainedStep returns either EOS or a parser-valid
//   next token; AppendConstrainedToken extends currentConstrainedOut by one valid
//   token, preserving parser.IsValidPrefix(currentConstrainedOut). CloseConstrainedSpan
//   sets insideConstrainedOut := false, making the implication vacuously true.
// progress:
//   OpenConstrainedSpan costs 1 step and appends "<<" to generated (length +1 = budget +1).
//   Each loop iteration costs exactly 1 step (AdaptiveConstrainedStep) and appends at most
//   one visible token, so |generated| <= |generatedPrefix| + steps is preserved.
//   CloseConstrainedSpan costs 1 step and appends ">>" (or nothing if already present),
//   so the length bound holds. Total cost = steps <= maxSteps.
// CSD_PROOF_SKETCH_END
generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

var steps: nat := 0;

helpers.AppendTaskGuidance(lm, "Generate exactly one valid SMILES string for an isocyanate molecule. Isocyanates contain the functional group R-N=C=O. Output only the SMILES string with no explanation, no Markdown, no extra text.");

// Phase 1: Open constrained span if not already inside one
if !insideConstrainedOut && steps < maxSteps {
  var og, oi, oc := helpers.OpenConstrainedSpan(lm, generated);
  generated := og;
  insideConstrainedOut := oi;
  currentConstrainedOut := oc;
  steps := steps + 1;
}

// Phase 2: Generate SMILES tokens inside constrained span
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
    // Close the span - we have a valid complete SMILES
    if steps < maxSteps {
      var cg, ci, cc := helpers.CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut);
      generated := cg;
      insideConstrainedOut := ci;
      currentConstrainedOut := cc;
      steps := steps + 1;
    } else {
      break;
    }
  } else {
    // Generate next SMILES token with adaptive constrained step + repetition penalty
    var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];
    var next := helpers.SafeRepetitionPenaltyStep(
      lm, parser, constrainedPrompt, currentConstrainedOut, generated, 1.5, eosToken
    );
    steps := steps + 1;
    if next == eosToken {
      // EOS before complete: check if we can close anyway
      if parser.IsCompletePrefix(currentConstrainedOut) && steps < maxSteps {
        var cg, ci, cc := helpers.CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut);
        generated := cg;
        insideConstrainedOut := ci;
        currentConstrainedOut := cc;
        steps := steps + 1;
      }
      break;
    } else {
      var ag, ai, ac := helpers.AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, next);
      generated := ag;
      insideConstrainedOut := ai;
      currentConstrainedOut := ac;
    }
  }
}

// Phase 3: If still inside span and budget remains, try CloseSpanWithinBudget
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
