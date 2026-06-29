// CSD_RATIONALE_BEGIN
// The key verification failure is that AppendConstrainedToken requires
// !parser.IsCompletePrefix(currentConstrained). In the "below minimum length"
// branch, we call AppendConstrainedToken without checking this precondition.
//
// Fix: Before calling AppendConstrainedToken, check !parser.IsCompletePrefix(currentConstrainedOut).
// If IsCompletePrefix is true and we're below minLength, we should NOT close
// (we want to extend), but we also CANNOT append more tokens since the precondition fails.
// In that case, just break out and close.
//
// The SMILES parser likely allows extending complete prefixes (e.g., "C" -> "C=C"),
// but the Dafny contract says !IsCompletePrefix is required. So we must respect that.
//
// When IsCompletePrefix is true (span is done) and valid extension exists:
// If we can't append (precondition forbids it), we should close the span.
// If IsCompletePrefix is false and token is valid: we can append.
//
// This means our minimum-length guard against tiny spans won't work for SMILES
// where single atoms are complete. Instead, if we're complete but short, we close
// anyway (the evaluator will see a short SMILES). The task is to generate a valid
// acrylate, so we rely heavily on the guidance + AdaptiveConstrainedStep to steer
// the generation toward longer acrylate structures before it becomes complete.
//
// Strategy:
// 1. Provide strong guidance specifying multi-token acrylate SMILES
// 2. Open constrained span
// 3. Generate with AdaptiveConstrainedStep/SafeRepetitionPenaltyStep
// 4. Only call AppendConstrainedToken when !IsCompletePrefix (required by API)
// 5. Call CloseSpanIfComplete each step to close when ready
// CSD_RATIONALE_END

// CSD_PROOF_SKETCH_BEGIN
// parser_validity: insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
//   - Init: copied from precondition (or OpenConstrainedSpan sets currentConstrainedOut := [],
//     and parser.IsValidPrefix([]) holds by precondition). OK.
//   - AppendConstrainedToken call site: guarded by both IsTokenValidNext and
//     !parser.IsCompletePrefix(currentConstrainedOut), satisfying all preconditions.
//     The postcondition preserves parser.IsValidPrefix. OK.
//   - CloseSpanIfComplete: when closed=true, sets insideConstrainedOut := false,
//     making the implication vacuous, and clears currentConstrainedOut. OK.
//   - CloseConstrainedSpan: sets insideConstrainedOut := false. OK.
//   - RollbackConstrainedToComplete: postcondition states IsCompletePrefix or empty,
//     and IsCompletePrefix ==> IsValidPrefix. OK.
//
// progress: |generated| <= |generatedPrefix| + steps
//   - OpenConstrainedSpan: steps += 1, |generated| += 1. Balanced.
//   - Each loop iteration increments steps by 1 (generation call) plus optionally 1
//     (CloseSpanIfComplete). AppendConstrainedToken adds 1 token. CloseSpanIfComplete
//     adds at most 1 token (">>"). Each step-consuming call adds at most 1 token.
//     So |generated| <= |generatedPrefix| + steps is maintained. OK.
// CSD_PROOF_SKETCH_END

generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

var steps: nat := 0;

helpers.AppendTaskGuidance(lm, "Generate a SMILES for a novel acrylate molecule. Acrylates contain C=CC(=O)O or C=C(C)C(=O)O. Generate a complete multi-atom SMILES like C=CC(=O)OCCC or C=C(C)C(=O)OCCCO or C=CC(=O)OCC(C)C. Output only the SMILES string.");

// Open constrained span if not already inside one
if steps < maxSteps && !insideConstrainedOut {
  var og, oi, oc := helpers.OpenConstrainedSpan(lm, generated);
  generated := og;
  insideConstrainedOut := oi;
  currentConstrainedOut := oc;
  steps := steps + 1;
}

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
    break;
  }

  // Generate next constrained token
  var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];
  var next := helpers.SafeRepetitionPenaltyStep(
    lm, parser, constrainedPrompt, currentConstrainedOut,
    generated, 2.0, eosToken
  );
  steps := steps + 1;

  if next == eosToken {
    // Rollback to complete prefix if possible
    var rg, rc := helpers.RollbackConstrainedToComplete(parser, generated, currentConstrainedOut);
    generated := rg;
    currentConstrainedOut := rc;
    // Close if complete and budget remains
    if parser.IsCompletePrefix(currentConstrainedOut) && steps < maxSteps {
      var fg, fi, fc := helpers.CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut);
      generated := fg;
      insideConstrainedOut := fi;
      currentConstrainedOut := fc;
      steps := steps + 1;
    }
    break;
  } else {
    // Only append if not yet complete (required by AppendConstrainedToken precondition)
    var isComplete := parser.IsCompletePrefix(currentConstrainedOut);
    var valid := helpers.IsTokenValidNext(parser, currentConstrainedOut, next);
    if valid && !isComplete {
      var ag, ai, ac := helpers.AppendConstrainedToken(
        lm, parser, generated, currentConstrainedOut, next
      );
      generated := ag;
      insideConstrainedOut := ai;
      currentConstrainedOut := ac;
    }

    // Try to close if complete (checks internally)
    if steps < maxSteps {
      var cg2, ci2, cc2, closed := helpers.CloseSpanIfComplete(
        lm, parser, generated, currentConstrainedOut
      );
      steps := steps + 1;
      if closed {
        generated := cg2;
        insideConstrainedOut := ci2;
        currentConstrainedOut := cc2;
        break;
      }
    }
  }
}

// Final cleanup: if still inside constrained span
if insideConstrainedOut && steps < maxSteps {
  var rg, rc := helpers.RollbackConstrainedToComplete(parser, generated, currentConstrainedOut);
  generated := rg;
  currentConstrainedOut := rc;
  if parser.IsCompletePrefix(currentConstrainedOut) && steps < maxSteps {
    var fg, fi, fc := helpers.CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut);
    generated := fg;
    insideConstrainedOut := fi;
    currentConstrainedOut := fc;
    steps := steps + 1;
  }
}

cost := steps;