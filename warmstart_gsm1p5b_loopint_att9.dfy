// CSD_RATIONALE_BEGIN
// Diagnosis from previous attempts:
// 1. The model naturally emits "<<" during unconstrained generation (reasoning phase).
//    When we detect "<<", we enter constrained mode via EnterObservedConstrainedSpan.
// 2. BUT: The model emits MULTIPLE << >> spans during reasoning (avg 3.8 per example).
//    The evaluator extracts the LAST visible span, which is often incomplete or malformed.
// 3. The main failure modes are:
//    a. malformed_constrained_content (22 examples): the constrained content has syntax errors
//    b. multiple spans cause confusion - the model keeps emitting << >> as part of reasoning
// 4. Key insight: We should CLOSE the constrained span as soon as it becomes a complete parse,
//    then EXIT constrained mode and CONTINUE generating freely. But we should also prevent
//    the model from re-entering constrained mode for intermediate reasoning spans, OR we
//    accept all spans but the constrained decoder ensures each one is syntactically valid.
// 5. Strategy: Accept all natural << emissions, enforce hard parser control for content,
//    close as soon as parser accepts, then continue freely. This gives many valid spans.
//    The LAST complete span is extracted as the answer.
// 6. Use AdaptiveConstrainedStep for better token selection inside spans.
// 7. Use RollbackConstrainedToComplete when EOS occurs inside a span to fix partial spans.
// 8. Budget management: the constrained phase needs enough steps. Don't force open too early.
// 9. Key fix from attempt 4 (best syntax at 57.1%): react to natural "<<" emission.
//    The issue is that constrained content often contains variables like {n} which the
//    arithmetic parser likely rejects. We need a parser that accepts variable expressions
//    or we need to guide the model to output only numeric expressions.
// 10. Better guidance: be very explicit that << >> must contain ONLY a numeric expression
//     with actual numbers (not variable names). This is the core issue.
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity: insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
//   - Unconstrained branch (next != "<<"): insideConstrainedOut stays false, implication vacuous.
//   - Unconstrained branch (next == "<<"): EnterObservedConstrainedSpan sets currentConstrainedOut
//     := [], which satisfies parser.IsValidPrefix([]) by method precondition. Invariant holds.
//   - Constrained branch, CloseSpanIfComplete (closed=true): insideConstrainedOut becomes false,
//     making the implication vacuously true; currentConstrainedOut := [].
//   - Constrained branch, CloseSpanIfComplete (closed=false): state unchanged, invariant holds
//     by loop invariant for the open span.
//   - Constrained branch, AdaptiveConstrainedStep returns non-EOS: AppendConstrainedToken
//     extends currentConstrainedOut with a parser-valid next token, preserving IsValidPrefix.
//   - Constrained branch, EOS: RollbackConstrainedToComplete produces a complete or empty
//     prefix satisfying IsValidPrefix. Then insideConstrainedOut := false makes it vacuous.
//
// progress: |generated| <= |generatedPrefix| + steps
//   - Each iteration increments steps by exactly 1 (only one step-consuming helper per branch).
//   - Unconstrained branch: steps+1, generated grows by 0 (EOS) or 1 (non-EOS token). OK.
//   - EnterObservedConstrainedSpan costs 0 steps, leaves generated unchanged. OK.
//   - Constrained CloseSpanIfComplete: steps+1, generated grows by at most 1 (">>" delimiter). OK.
//   - Constrained AdaptiveConstrainedStep: steps+1, generated grows by at most 1 via
//     AppendConstrainedToken. EOS path: steps+1, generated unchanged (rollback may shrink). OK.
// CSD_PROOF_SKETCH_END
generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

helpers.AppendTaskGuidance(lm, "Solve this math word problem step by step. At the very end of your response, write the final answer as a single arithmetic expression using ONLY actual numbers (like 5, 12.5, etc.) and operators (+, -, *, /, (, )). Put this expression between << and >>. Example: <<5 * 3 + 2>>. Do not use variable names or words inside << >>.");

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
    var next := helpers.UnconstrainedStep(lm, prompt, generated);
    steps := steps + 1;
    if next == eosToken {
      break;
    } else {
      generated := generated + [next];
      if next == "<<" {
        // Model opened a constrained span - enter constrained mode
        var og, oi, oc := helpers.EnterObservedConstrainedSpan(lm, generated);
        generated := og;
        insideConstrainedOut := oi;
        currentConstrainedOut := oc;
      }
    }
  } else {
    // Inside constrained span
    // First check if current content is already complete - if so, close immediately
    var cg, ci, cc, closed := helpers.CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut);
    steps := steps + 1;
    if closed {
      generated := cg;
      insideConstrainedOut := ci;
      currentConstrainedOut := cc;
    } else {
      // Not complete yet - need to generate more content
      // But we already consumed a step with CloseSpanIfComplete, so check budget
      if steps >= maxSteps {
        // No budget left - rollback to complete and exit
        var rg, rc := helpers.RollbackConstrainedToComplete(parser, generated, currentConstrainedOut);
        generated := rg;
        currentConstrainedOut := rc;
        insideConstrainedOut := false;
        currentConstrainedOut := [];
        break;
      }
      // Generate next constrained token using adaptive step with group awareness
      var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];
      var next := helpers.AdaptiveConstrainedStep(lm, parser, constrainedPrompt, currentConstrainedOut, validTokenGroups, 4.0, 12, eosToken);
      steps := steps + 1;
      if next == eosToken {
        // EOS inside constrained span - rollback to complete and exit
        var rg, rc := helpers.RollbackConstrainedToComplete(parser, generated, currentConstrainedOut);
        generated := rg;
        currentConstrainedOut := rc;
        insideConstrainedOut := false;
        currentConstrainedOut := [];
        break;
      } else {
        var ag, ai, ac := helpers.AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, next);
        generated := ag;
        insideConstrainedOut := ai;
        currentConstrainedOut := ac;
      }
    }
  }
}

cost := steps;
