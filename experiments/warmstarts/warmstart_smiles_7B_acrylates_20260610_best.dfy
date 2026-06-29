// CSD_RATIONALE_BEGIN
// The verification error is that AppendConstrainedToken requires
// !parser.IsCompletePrefix(currentConstrained) but we haven't checked this
// before calling it. After receiving a non-EOS token from AdaptiveConstrainedStep
// or similar, the current constrained prefix might already be complete (e.g., "C"),
// and appending would violate the precondition.
//
// Fix: Before calling AppendConstrainedToken, check that the current prefix is NOT
// complete. If it IS already complete and we got a non-EOS token, we should close
// the span instead of appending.
//
// Also need: next in lm.Tokens and parser.IsValidPrefix(currentConstrainedOut + [next]).
// The ConstrainedStep/AdaptiveConstrainedStep helpers guarantee the token is parser-valid,
// but Dafny needs a static check. We use IsTokenValidNext to verify validity before
// calling AppendConstrainedToken.
//
// Strategy: 
// 1. Open constrained span
// 2. Loop: AdaptiveConstrainedStep for token selection
// 3. Before AppendConstrainedToken: check !parser.IsCompletePrefix(currentConstrainedOut)
//    AND helpers.IsTokenValidNext(parser, currentConstrainedOut, next)
// 4. If already complete before appending: close the span
// 5. On EOS: rollback to complete and close
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity: insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
//   - OpenConstrainedSpan: sets currentConstrainedOut := [], insideConstrainedOut := true.
//     parser.IsValidPrefix([]) holds by precondition. Established.
//   - AppendConstrainedToken branch: we check IsTokenValidNext(parser, currentConstrainedOut, next)
//     before calling, so parser.IsValidPrefix(currentConstrainedOut + [next]) holds.
//     AppendConstrainedToken preserves IsValidPrefix by this check.
//   - CloseConstrainedSpan: sets insideConstrainedOut := false, making implication vacuous.
//   - RollbackConstrainedToComplete: preserves IsValidPrefix (returns valid or empty prefix).
//   - canClose/EOS branches that close: insideConstrainedOut becomes false, vacuous.
//
// progress: |generated| <= |generatedPrefix| + steps
//   - OpenConstrainedSpan: steps += 1, |generated| += 1. Maintained.
//   - AdaptiveConstrainedStep costs 1 step, steps += 1, |generated| grows by at most 1
//     (AppendConstrainedToken adds one token). Maintained.
//   - CloseConstrainedSpan: steps += 1, |generated| += 1. Maintained.
//   - CloseSpanIfComplete: steps += 1, |generated| += at most 1. Maintained.
//   - All other operations (Rollback, IsTokenValidNext) are cost 0. Maintained.
// CSD_PROOF_SKETCH_END
generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

var guidance: string := "Generate a SMILES string for an acrylate molecule. The acrylate MUST contain the acryloyl group: a vinyl group connected to an ester. Valid examples: C=CC(=O)OCC, C=CC(=O)OCCC, C=CC(=O)OC, C=C(C)C(=O)OCC. Start with bracket atom or multi-char prefix to avoid trivial single-atom completions.";
helpers.AppendTaskGuidance(lm, guidance);

var steps: nat := 0;

var acrylateTokens: seq<Token> := ["C", "=", "(", ")", "O", "c", "n", "N", "S", "s", "o"];
var vinylTokens: seq<Token> := ["=", "C", "("];
var esterTokens: seq<Token> := ["O", "(", "=", ")"];
var acrylateGroups: seq<seq<Token>> := [acrylateTokens, vinylTokens, esterTokens];

var minLength: nat := 8;

// Enter constrained span if not already inside
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

  // Check if we can close (minimum length AND complete)
  var isComplete := parser.IsCompletePrefix(currentConstrainedOut);
  var lenOk := |currentConstrainedOut| >= minLength;

  if isComplete && lenOk {
    var cg, ci, cc := helpers.CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut);
    generated := cg;
    insideConstrainedOut := ci;
    currentConstrainedOut := cc;
    steps := steps + 1;
    break;
  }

  // Generate next constrained token
  var stableLen := |generated| - |currentConstrainedOut|;
  var constrainedPrompt := prompt + generated[..stableLen];
  var combined: seq<seq<Token>> := validTokenGroups + acrylateGroups;

  var next: Token;
  var tokenCount := |currentConstrainedOut|;
  if tokenCount == 0 {
    var startGroups: seq<seq<Token>> := combined + [["C", "="]];
    next := helpers.AdaptiveConstrainedStepWithPenalties(
      lm, parser, constrainedPrompt, currentConstrainedOut,
      startGroups, 10.0, ["N", "O", "S", "F", "B", "P", "I", "[", "c", "n", "o", "s"], 6.0, 100, eosToken
    );
  } else if tokenCount < minLength {
    next := helpers.AdaptiveConstrainedStep(
      lm, parser, constrainedPrompt, currentConstrainedOut,
      combined, 6.0, 50, eosToken
    );
  } else {
    next := helpers.SafeRepetitionPenaltyStep(
      lm, parser, constrainedPrompt, currentConstrainedOut,
      generated, 2.0, eosToken
    );
  }

  steps := steps + 1;

  if next == eosToken {
    // EOS: rollback to complete and close if possible
    var rg, rc := helpers.RollbackConstrainedToComplete(parser, generated, currentConstrainedOut);
    generated := rg;
    currentConstrainedOut := rc;
    var rComplete := parser.IsCompletePrefix(currentConstrainedOut);
    if rComplete && steps < maxSteps {
      var cg, ci, cc := helpers.CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut);
      generated := cg;
      insideConstrainedOut := ci;
      currentConstrainedOut := cc;
      steps := steps + 1;
    }
    break;
  } else {
    // Check preconditions for AppendConstrainedToken:
    // 1. !parser.IsCompletePrefix(currentConstrainedOut)
    // 2. parser.IsValidPrefix(currentConstrainedOut + [next])
    var curComplete := parser.IsCompletePrefix(currentConstrainedOut);
    if curComplete {
      // Already complete: close the span without appending
      if steps < maxSteps {
        var cg, ci, cc := helpers.CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut);
        generated := cg;
        insideConstrainedOut := ci;
        currentConstrainedOut := cc;
        steps := steps + 1;
      }
      break;
    } else {
      // Not complete: check if next token is valid
      var valid := helpers.IsTokenValidNext(parser, currentConstrainedOut, next);
      if valid {
        var ag, ai, ac := helpers.AppendConstrainedToken(
          lm, parser, generated, currentConstrainedOut, next
        );
        generated := ag;
        insideConstrainedOut := ai;
        currentConstrainedOut := ac;
      }
      // If not valid (shouldn't happen with constrained step), just skip
    }
  }
}

// Final cleanup: if still inside, rollback and close
if insideConstrainedOut && steps < maxSteps {
  var rg, rc := helpers.RollbackConstrainedToComplete(parser, generated, currentConstrainedOut);
  generated := rg;
  currentConstrainedOut := rc;
  var rComplete := parser.IsCompletePrefix(currentConstrainedOut);
  if rComplete && steps < maxSteps {
    var cg, ci, cc := helpers.CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut);
    generated := cg;
    insideConstrainedOut := ci;
    currentConstrainedOut := cc;
    steps := steps + 1;
  }
}

cost := steps;