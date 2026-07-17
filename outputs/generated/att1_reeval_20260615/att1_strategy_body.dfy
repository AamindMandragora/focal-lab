// CSD_RATIONALE_BEGIN
// SQL-generation CSD for the Spider dataset. The task requires producing output
// in the form "SQL: <<YOUR QUERY>>". Strategy:
// 1. Append task guidance to steer the model toward SQL generation.
// 2. Use UnconstrainedStep to generate "SQL: " prefix freely until "<<" is observed.
// 3. Once inside the constrained span, use AdaptiveConstrainedStep with the
//    caller-supplied validTokenGroups (which should contain SQL keywords/identifiers
//    from the schema context) to generate a valid SQL query token by token.
// 4. Use CloseSpanIfComplete to close the span with ">>" once the parser accepts
//    a complete SQL expression.
// 5. After the span closes, allow a few more unconstrained tokens for any trailing
//    content, then stop.
//
// State tracked:
// - generated: full output including "SQL: <<...>>"
// - insideConstrainedOut: whether we are inside the constrained SQL span
// - currentConstrainedOut: the SQL tokens generated so far (parser-tracked)
// - steps/cost: token budget consumed
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity:
//   1. Outside span (free generation): implication is vacuously true since
//      insideConstrainedOut is false. When next == "<<", we set
//      currentConstrainedOut := [] which is valid by parser.IsValidPrefix([])
//      (a precondition of the method).
//   2. CloseSpanIfComplete: when closed==true, insideConstrainedOut becomes false
//      (implication vacuous); when closed==false, state is unchanged, so the
//      invariant is trivially preserved.
//   3. AdaptiveConstrainedStep: by its contract, returns either eosToken or a
//      token t such that parser.IsValidPrefix(currentConstrainedOut + [t]).
//      AppendConstrainedToken then sets currentConstrainedOut := currentConstrainedOut + [t],
//      preserving parser_validity.
//   4. After span close (free trailing): insideConstrainedOut is false, so the
//      implication is vacuous.
//
// progress (|generated| <= |generatedPrefix| + steps):
//   1. Free generation branch: steps += 1, generated grows by at most 1 token
//      (the appended next token, or 0 on EOS). Bound preserved.
//   2. CloseSpanIfComplete branch: steps += 1, generated grows by at most 1
//      token (the ">>" delimiter when closing). Bound preserved.
//   3. Constrained step branch: steps += 1, AppendConstrainedToken adds exactly
//      1 token to generated (or 0 on EOS break). Bound preserved.
//   4. Post-span free trailing: steps += 1, generated grows by at most 1 token.
//      Bound preserved.
// CSD_PROOF_SKETCH_END
generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

helpers.AppendTaskGuidance(lm, "Generate a valid SQL query for the given database schema. Output format: SQL: <<your SQL query here>>. Use only table and column names from the provided schema. Do not add explanations.");

var steps: nat := 0;

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
    // Free generation until we observe "<<" opening the constrained SQL span
    var next := helpers.UnconstrainedStep(lm, prompt, generated);
    steps := steps + 1;
    if next == eosToken {
      break;
    }
    generated := generated + [next];
    if next == "<<" {
      insideConstrainedOut := true;
      currentConstrainedOut := [];
    }
  } else {
    // Inside constrained SQL span
    var cg, ci, cc, closed := helpers.CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut);
    steps := steps + 1;
    if closed {
      generated := cg;
      insideConstrainedOut := ci;
      currentConstrainedOut := cc;
      // Span is now closed; exit the loop
      break;
    } else {
      // Not yet complete: generate next SQL token using adaptive constrained step
      // with schema-grounded token groups to prefer valid SQL identifiers/keywords
      var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];
      var next := helpers.AdaptiveConstrainedStep(
        lm, parser, constrainedPrompt, currentConstrainedOut,
        validTokenGroups, 4.0, 12, eosToken
      );
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
}

cost := steps;
