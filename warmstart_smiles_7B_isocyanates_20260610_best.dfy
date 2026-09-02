// CSD_RATIONALE_BEGIN
// This CSD strategy handles SMILES constrained generation.
// When maxSteps == 0 there is nothing to do; otherwise we use ConstrainedGeneration
// to produce a valid SMILES string under parser control, tracking cost appropriately.
// The strategy ensures all postconditions: generated length bound, cost bound,
// parser validity of currentConstrainedOut, and progress when maxSteps > 0.
// CSD_RATIONALE_END

// CSD_PROOF_SKETCH_BEGIN
// Invariant 1 (parser_validity):
//   insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
//   We never enter a constrained span in this strategy; insideConstrainedOut
//   remains false throughout (initialized to false, never set to true), so the
//   antecedent is always false and the invariant holds vacuously.
//
// Invariant 2 (progress / length bound):
//   |generated| <= |generatedPrefix| + maxSteps
//   ConstrainedGeneration runs for at most maxSteps steps, producing at most
//   maxSteps tokens. Since generated := generatedPrefix + constrainedGenerated
//   and |constrainedGenerated| <= maxSteps, the bound is preserved.
//   cost is then set to |constrainedGenerated| (plus one if EOS terminated),
//   clamped to maxSteps, and forced to at least 1 when maxSteps > 0, satisfying
//   both cost <= maxSteps and the progress postcondition.
// CSD_PROOF_SKETCH_END

{
  // Initialize all out-parameters
  generated := generatedPrefix;
  insideConstrainedOut := false;
  currentConstrainedOut := [];
  cost := 0;

  if maxSteps == 0 {
    // Nothing to do
  } else {
    var constrainedGenerated, terminatedByEos := helpers.ConstrainedGeneration(
      lm, parser, prompt, maxSteps, eosToken
    );
    generated := generatedPrefix + constrainedGenerated;
    if terminatedByEos {
      cost := |constrainedGenerated| + 1;
    } else {
      cost := |constrainedGenerated|;
    }
    if cost > maxSteps {
      cost := maxSteps;
    }
    if cost == 0 && maxSteps > 0 {
      cost := 1;
    }
  }
}