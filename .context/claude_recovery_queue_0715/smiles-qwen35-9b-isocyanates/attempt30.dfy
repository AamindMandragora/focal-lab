// CSD_RATIONALE_BEGIN
// SMILES isocyanate CSD - improving from 20% accuracy baseline.
//
// Analysis of current best (20% accuracy, 100% syntax):
// - The strategy works syntactically (100% syntax rate)
// - But only 20% accuracy means ~10/50 unique valid isocyanate molecules
// - The failing examples show "O=C=Nc>>" - incomplete SMILES (aromatic c without ring closure)
//   or other partial structures
//
// Root cause of 0% in last attempt:
// - The previous attempt (26, 29) achieved 0% syntax - likely due to the span being opened
//   but the output containing "O=C=Nc>>" which is syntactically invalid SMILES
// - The issue is that when minSpanLength check prevents closure and EOS is hit, the span
//   remains open and CloseSpanWithinBudget may produce invalid SMILES
//
// Strategy to improve from 20% to 41%+:
// 1. Keep the working approach from best attempt (attempt 9/best)
// 2. Increase diversity by using more aggressive temperature variation
// 3. Use SafePenalizedConstrainedStep to penalize the first few common tokens
// 4. Lower temperature gradually to help complete valid structures
// 5. Remove the NCO content check that may cause the span to remain open (let parser decide)
// 6. Use CloseSpanIfComplete to close as soon as a valid parse is available
// 7. Keep minSpanLength = 5 to ensure non-trivial molecules
//
// Key insight from failures: The "O=C=Nc" problem suggests the model starts with
// aromatic ring fragments. We need to guide it to produce complete valid SMILES.
// The guidance should show the canonical NCO format with complete examples.
//
// The guidance worked well - the model learned "O=C=N..." prefix structure.
// The problem is that after "O=C=Nc" (which starts an aromatic ring), the model
// sometimes emits EOS before closing the ring, giving invalid SMILES.
//
// Fix: Use CloseSpanWithinBudget more aggressively to complete partial structures.
// Don't check NCO content (let parser handle validity of complete structures).
// Ensure we always call CloseSpanWithinBudget at the end if still inside.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity:
//   OpenConstrainedSpan sets insideConstrainedOut := true and currentConstrainedOut := [],
//   which satisfies parser.IsValidPrefix([]) by precondition.
//   Each constrained step (SafeTemperatureConstrainedStep, SafePenalizedConstrainedStep,
//   GroupBoostedConstrainedStep) returns either EOS or a parser-valid next token.
//   AppendConstrainedToken preserves parser.IsValidPrefix by its postcondition.
//   CloseConstrainedSpan sets insideConstrainedOut := false (implication vacuously true,
//   currentConstrainedOut := []).
//   CloseSpanIfComplete: if closed, sets insideConstrainedOut := false; if not closed, 
//   leaves state unchanged (still parser-valid).
//   CloseSpanWithinBudget: postcondition guarantees insideOut ==> parser.IsValidPrefix(currentOut).
// progress:
//   OpenConstrainedSpan: +1 step (|generated| grows by 1 for "<<").
//   Each loop iteration: exactly 1 step helper call costs 1 step,
//   AppendConstrainedToken adds at most 1 visible token.
//   CloseSpanIfComplete costs +1 step (either closes or is no-op, but still costs 1).
//   CloseSpanWithinBudget: costs <= closeBudget = maxSteps - steps, so total <= maxSteps.
//   |generated| <= |generatedPrefix| + steps at all times.
//   Total cost = steps <= maxSteps always.
// CSD_PROOF_SKETCH_END

generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

var steps: nat := 0;

helpers.AppendTaskGuidance(lm, "Generate one valid SMILES for a molecule in the isocyanate class (R-N=C=O). Examples of valid isocyanate SMILES: O=C=NC (methyl isocyanate), O=C=NCC (ethyl isocyanate), O=C=NCCC (propyl isocyanate), O=C=NCCCC (butyl isocyanate), O=C=NC(C)C (isopropyl isocyanate), O=C=NC(C)(C)C (tert-butyl isocyanate), O=C=NC1CC1 (cyclopropyl isocyanate), O=C=NC1CCCC1 (cyclopentyl isocyanate), O=C=NC1CCCCC1 (cyclohexyl isocyanate), O=C=NCc1ccccc1 (benzyl isocyanate), O=C=Nc1ccccc1 (phenyl isocyanate), O=C=Nc1ccc(C)cc1 (4-tolyl isocyanate), O=C=Nc1ccc(Cl)cc1 (4-chlorophenyl isocyanate), O=C=Nc1ccc(F)cc1 (4-fluorophenyl isocyanate), O=C=Nc1ccccn1 (3-pyridyl isocyanate), O=C=NCC=C (allyl isocyanate), O=C=NCC#C (propargyl isocyanate), O=C=NCF (fluoromethyl isocyanate), O=C=NCCF (2-fluoroethyl isocyanate). Output ONLY the SMILES string with no other text.");

// Open constrained span if not already inside one
if !insideConstrainedOut && steps < maxSteps {
  var og, oi, oc := helpers.OpenConstrainedSpan(lm, generated);
  generated := og;
  insideConstrainedOut := oi;
  currentConstrainedOut := oc;
  steps := steps + 1;
}

// Token count for phase control
var tokenCount: nat := 0;

// Common tokens to penalize early to avoid degeneracy
var commonTokens: seq<Token> := ["C", "CC", "CCC", "O", "N"];

// Main generation loop
while steps < maxSteps && insideConstrainedOut
  invariant 0 <= steps <= maxSteps
  invariant lm.ValidTokensIdsLogits()
  invariant !insideConstrainedOut ==> currentConstrainedOut == []
  invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
  invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|
  invariant |generated| <= |generatedPrefix| + steps
  decreases maxSteps - steps
{
  // Try to close if complete and has enough tokens
  if parser.IsCompletePrefix(currentConstrainedOut) && |currentConstrainedOut| >= 5 {
    var cg, ci, cc := helpers.CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut);
    generated := cg;
    insideConstrainedOut := ci;
    currentConstrainedOut := cc;
    steps := steps + 1;
  } else {
    var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];
    var next := eosToken;

    if tokenCount < 4 {
      // Very early tokens: penalize common simple tokens + high temperature for diversity
      next := helpers.SafePenalizedConstrainedStep(
        lm, parser, constrainedPrompt, currentConstrainedOut, commonTokens, 5.0, eosToken
      );
    } else if tokenCount < 12 {
      // Early-mid phase: high temperature for structural diversity
      next := helpers.SafeTemperatureConstrainedStep(
        lm, parser, constrainedPrompt, currentConstrainedOut, 1.8, eosToken
      );
    } else if tokenCount < 25 {
      // Mid phase: moderate temperature 
      next := helpers.SafeTemperatureConstrainedStep(
        lm, parser, constrainedPrompt, currentConstrainedOut, 1.5, eosToken
      );
    } else {
      // Late phase: group-boosted to complete the structure
      next := helpers.GroupBoostedConstrainedStep(
        lm, parser, constrainedPrompt, currentConstrainedOut, validTokenGroups, 4.0, eosToken
      );
    }

    steps := steps + 1;
    if next == eosToken {
      break;
    } else {
      var ag, ai, ac := helpers.AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, next);
      generated := ag;
      insideConstrainedOut := ai;
      currentConstrainedOut := ac;
      tokenCount := tokenCount + 1;
    }
  }
}

// If still inside span and budget remains, use CloseSpanWithinBudget
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
