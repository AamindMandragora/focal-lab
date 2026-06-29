// CSD_RATIONALE_BEGIN
// Root cause from attempt 7: The model emits "SQL: <<" as its very first token
// (avg 1 token before first visible open). Then the strategy entered constrained
// mode but ran out of budget because UnconstrainedChunk consumed tokens AND the
// span content was never generated (only 4 tokens generated avg: "SQL", ":", "<<",
// and one constrained attempt before budget ran out).
//
// The best result (attempt 2/6) used a simple single-step loop: unconstrained
// steps until "<<" is observed, then constrained steps with CloseSpanIfComplete.
// That achieved 63-64% accuracy and 98-99% syntax.
//
// The remaining 36% failure is semantic (wrong SQL query). The feedback says
// "syntax_valid_semantic_mismatch=34-36" — the SQL is syntactically valid but
// semantically wrong. This is a content quality issue, not a structural issue.
//
// Strategy to improve from 64% to 68%+:
// 1. Keep the working structure from attempts 2/6 (simple loop, unconstrained
//    until "<<", then constrained with CloseSpanIfComplete).
// 2. Add stronger guidance to help the model pick the right tables/columns.
// 3. Use GroupBoostedConstrainedStep instead of AdaptiveConstrainedStep to
//    leverage validTokenGroups (which contains schema-relevant tokens) more
//    aggressively inside the span.
// 4. Use SafeRepetitionPenaltyStep variant to avoid repeating tokens.
// 5. Keep budget management tight: the SQL query is ~20-80 tokens, and we have
//    1200 steps total. The unconstrained prefix takes ~1 token, leaving ~1199
//    for the constrained SQL body.
//
// Key structural note: The model outputs "SQL: <<" as a single compound token
// OR as separate tokens. Either way, after the "<<" token appears in generated,
// we enter constrained mode. The simple approach from attempts 2/6 works.
//
// The main improvement levers:
// - Use GroupBoostedConstrainedStep with higher boost (6.0) to prefer tokens
//   from validTokenGroups (schema tables/columns).
// - Keep CloseSpanIfComplete as the first check each constrained iteration.
// - Ensure cost > 0 when maxSteps > 0 (the postcondition requirement).
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// parser_validity invariant:
//   Outside the span (!insideConstrainedOut): when next == "<<", we set
//   currentConstrainedOut := [], which satisfies parser.IsValidPrefix([])
//   by precondition. Otherwise the implication is vacuously true.
//   Inside the span: CloseSpanIfComplete either closes (setting
//   insideConstrainedOut := false, making the implication vacuous) or is
//   a no-op (cost +0, state unchanged). In the no-op path, GroupBoostedConstrainedStep
//   returns either EOS (we break) or a parser-valid next token; AppendConstrainedToken
//   extends currentConstrainedOut with that valid token, preserving
//   parser.IsValidPrefix(currentConstrainedOut) by the contract of AppendConstrainedToken.
//
// progress invariant (|generated| <= |generatedPrefix| + steps):
//   Every loop iteration increments steps by exactly 1.
//   - Unconstrained branch: appends at most 1 token (or breaks on EOS),
//     so |generated| grows by at most 1 while steps grows by 1.
//   - CloseSpanIfComplete closed=true: steps += 1, appends ">>" (1 token)
//     and we break. |generated| grows by 1, steps grows by 1.
//   - CloseSpanIfComplete closed=false (cost +0) + GroupBoostedConstrainedStep
//     (cost +1): steps += 1 total for the combined operation. If EOS, no
//     append; if valid token, AppendConstrainedToken appends 1 token.
//     |generated| grows by at most 1, steps grows by 1. Invariant preserved.
// CSD_PROOF_SKETCH_END

generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

var guidance: string := "Answer with exactly: SQL: <<your_sql_query>>. Output only that single line. No explanation. No markdown. The complete SQL query goes between << and >>. Use the schema provided. Example: SQL: <<SELECT col FROM tbl WHERE cond>>";
helpers.AppendTaskGuidance(lm, guidance);

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
        insideConstrainedOut := true;
        currentConstrainedOut := [];
      }
    }
  } else {
    // Inside constrained span: check if we can close first (cost +0 if not closed)
    var cg, ci, cc, closed := helpers.CloseSpanIfComplete(lm, parser, generated, currentConstrainedOut);
    if closed {
      steps := steps + 1;
      generated := cg;
      insideConstrainedOut := ci;
      currentConstrainedOut := cc;
      break;
    } else {
      // Generate next constrained token using group-boosted step to prefer schema tokens
      var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];
      var next := helpers.GroupBoostedConstrainedStep(
        lm, parser, constrainedPrompt, currentConstrainedOut,
        validTokenGroups, 6.0, eosToken
      );
      steps := steps + 1;
      if next == eosToken {
        // EOS before complete: rollback to last complete prefix and close if possible
        var rg, rc := helpers.RollbackConstrainedToComplete(parser, generated, currentConstrainedOut);
        generated := rg;
        currentConstrainedOut := rc;
        insideConstrainedOut := true;
        if parser.IsCompletePrefix(currentConstrainedOut) && steps < maxSteps {
          var cg2, ci2, cc2 := helpers.CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut);
          steps := steps + 1;
          generated := cg2;
          insideConstrainedOut := ci2;
          currentConstrainedOut := cc2;
        }
        break;
      } else {
        var ag, ai, ac := helpers.AppendConstrainedToken(
          lm, parser, generated, currentConstrainedOut, next
        );
        generated := ag;
        insideConstrainedOut := ai;
        currentConstrainedOut := ac;
      }
    }
  }
}

// Post-loop: if still inside span and it's complete and budget remains, close it
if insideConstrainedOut && parser.IsCompletePrefix(currentConstrainedOut) && steps < maxSteps {
  var cg, ci, cc := helpers.CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut);
  steps := steps + 1;
  generated := cg;
  insideConstrainedOut := ci;
  currentConstrainedOut := cc;
}

cost := steps;