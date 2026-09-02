    // CSD_RATIONALE_BEGIN
// Strategy: Unconstrained generation with reactive << trigger.
// When << is detected, enter constrained mode.
// Inside constrained spans:
//   - If parser.IsCompletePrefix: close the span immediately.
//   - If span exceeds maxSpanTokens (35): rollback to last complete, then close or exit.
//   - If validTokenCount <= narrowThreshold (12): use GroupBoostedConstrainedStep (boost=5.0)
//     to force valid completions when the grammar is nearly determined.
//   - Otherwise: use SafeRepetitionPenaltyStep (penalty=4.0) for broader exploration
//     while discouraging repetition.
// On EOS inside constrained: rollback+close if complete, else exit constrained.
// CSD_RATIONALE_END
generated := generatedPrefix;
insideConstrainedOut := insideConstrained;
currentConstrainedOut := currentConstrained;
cost := 0;

helpers.AppendTaskGuidance(lm, "Solve the math word problem step by step. For each calculation and for the final answer, write a short arithmetic expression inside << >> delimiters, e.g. <<3+5=8>>. Keep each expression brief and complete.");

var steps: nat := 0;
var narrowThreshold: nat := 12;
var maxSpanTokens: nat := 35;
var spanTokens: nat := 0;

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
        spanTokens := 0;
      }
    }
  } else if parser.IsCompletePrefix(currentConstrainedOut) {
    var cg, ci, cc := helpers.CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut);
    generated := cg;
    insideConstrainedOut := ci;
    currentConstrainedOut := cc;
    steps := steps + 1;
    spanTokens := 0;
  } else if spanTokens >= maxSpanTokens {
    var rg, rc := helpers.RollbackConstrainedToComplete(parser, generated, currentConstrainedOut);
    generated := rg;
    currentConstrainedOut := rc;
    if parser.IsCompletePrefix(currentConstrainedOut) {
      var cg2, ci2, cc2 := helpers.CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut);
      generated := cg2;
      insideConstrainedOut := ci2;
      currentConstrainedOut := cc2;
      steps := steps + 1;
    } else {
      insideConstrainedOut := false;
      currentConstrainedOut := [];
      if steps < maxSteps {
        var next2 := helpers.UnconstrainedStep(lm, prompt, generated);
        steps := steps + 1;
        if next2 == eosToken {
          break;
        } else {
          generated := generated + [next2];
          if next2 == "<<" {
            insideConstrainedOut := true;
            currentConstrainedOut := [];
            spanTokens := 0;
          }
        }
      }
    }
    spanTokens := 0;
  } else {
    var constrainedPrompt := prompt + generated[..|generated| - |currentConstrainedOut|];
    var validCount := helpers.ValidTokenCount(parser, currentConstrainedOut);
    var next3: Token;
    if validCount <= narrowThreshold {
      next3 := helpers.GroupBoostedConstrainedStep(lm, parser, constrainedPrompt, currentConstrainedOut, validTokenGroups, 5.0, eosToken);
    } else {
      next3 := helpers.SafeRepetitionPenaltyStep(lm, parser, constrainedPrompt, currentConstrainedOut, generated, 4.0, eosToken);
    }
    steps := steps + 1;
    spanTokens := spanTokens + 1;
    if next3 == eosToken {
      if parser.IsCompletePrefix(currentConstrainedOut) {
        if steps < maxSteps {
          var cg3, ci3, cc3 := helpers.CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut);
          generated := cg3;
          insideConstrainedOut := ci3;
          currentConstrainedOut := cc3;
          steps := steps + 1;
        }
      } else {
        var rg2, rc2 := helpers.RollbackConstrainedToComplete(parser, generated, currentConstrainedOut);
        generated := rg2;
        currentConstrainedOut := rc2;
        if parser.IsCompletePrefix(currentConstrainedOut) && steps < maxSteps {
          var cg4, ci4, cc4 := helpers.CloseConstrainedSpan(lm, parser, generated, currentConstrainedOut);
          generated := cg4;
          insideConstrainedOut := ci4;
          currentConstrainedOut := cc4;
          steps := steps + 1;
        } else {
          insideConstrainedOut := false;
          currentConstrainedOut := [];
        }
      }
      break;
    } else {
      var ag, ai, ac := helpers.AppendConstrainedToken(lm, parser, generated, currentConstrainedOut, next3);
      generated := ag;
      insideConstrainedOut := ai;
      currentConstrainedOut := ac;
    }
  }
}

cost := steps;
