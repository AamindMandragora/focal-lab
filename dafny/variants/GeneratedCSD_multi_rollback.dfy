include "../VerifiedAgentSynthesis.dfy"

module GeneratedCSD {
  import opened VerifiedDecoderAgent

  method MyCSDStrategy(
    lm: LM,
    parser: Parser,
    prompt: Prefix,
    generatedPrefix: Prefix,
    insideConstrained: bool,
    currentConstrained: Prefix,
    maxSteps: nat,
    stepTokenBudget: nat,
    validTokenGroups: seq<seq<Token>>,
    eosToken: Token
  ) returns (
    generated: Prefix,
    insideConstrainedOut: bool,
    currentConstrainedOut: Prefix,
    cost: int
  )
    modifies lm.Logits
    requires lm.ValidTokensIdsLogits()
    requires parser.IsValidPrefix([])
    requires !insideConstrained ==> currentConstrained == []
    requires insideConstrained ==> parser.IsValidPrefix(currentConstrained)
    requires insideConstrained ==> |currentConstrained| <= |generatedPrefix|
    requires insideConstrained ==> generatedPrefix[|generatedPrefix| - |currentConstrained|..] == currentConstrained
    requires "<<" in lm.Tokens && ">>" in lm.Tokens
    requires eosToken in lm.Tokens
    ensures lm.ValidTokensIdsLogits()
    ensures |generated| <= |generatedPrefix| + maxSteps
    ensures !insideConstrainedOut ==> currentConstrainedOut == []
    ensures insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
    ensures cost <= maxSteps
    ensures maxSteps == 0 || cost > 0 || generated != generatedPrefix ||
            insideConstrainedOut != insideConstrained ||
            currentConstrainedOut != currentConstrained

  {
    var helpers := new CSDHelpers();
    generated := generatedPrefix;
    insideConstrainedOut := insideConstrained;
    currentConstrainedOut := currentConstrained;
    cost := 0;

    var steps: nat := 0;

    while steps < maxSteps
      invariant 0 <= steps <= maxSteps
      invariant lm.ValidTokensIdsLogits()
      invariant !insideConstrainedOut ==> currentConstrainedOut == []
      invariant insideConstrainedOut ==> parser.IsValidPrefix(currentConstrainedOut)
      invariant insideConstrainedOut ==> |currentConstrainedOut| <= |generated|
      invariant insideConstrainedOut ==> generated[|generated| - |currentConstrainedOut|..] == currentConstrainedOut
      invariant |generated| <= |generatedPrefix| + steps
      invariant cost == 0
      decreases maxSteps - steps
    {
      if !insideConstrainedOut {
        var chunkBudget: nat := maxSteps - steps;
        var chunkedGenerated, stoppedOnOpenSpan, stoppedOnEos, stepsUsed := helpers.UnconstrainedChunk(
          lm, prompt, generated, chunkBudget, "<<", eosToken
        );
        generated := chunkedGenerated;
        steps := steps + stepsUsed;
        if stoppedOnEos {
          break;
        } else if stoppedOnOpenSpan {
          insideConstrainedOut := true;
          currentConstrainedOut := [];
        }
      } else {
        var isComplete := parser.IsCompletePrefix(currentConstrainedOut);
        if isComplete {
          var closedGenerated, closedInside, closedCurrent := helpers.CloseConstrainedSpan(
            lm, parser, generated, currentConstrainedOut
          );
          generated := closedGenerated;
          insideConstrainedOut := closedInside;
          currentConstrainedOut := closedCurrent;
          steps := steps + 1;
        } else {
          var deadEnd := helpers.DeadEndDetection(parser, currentConstrainedOut, 1);
          if deadEnd {
            // Try multiple rollback boundaries: comma, WHERE, FROM, JOIN, ON
            var repairedComma := helpers.RollbackToBoundary(parser, currentConstrainedOut, ",");
            var repairedWhere := helpers.RollbackToBoundary(parser, currentConstrainedOut, "WHERE");
            var repairedFrom := helpers.RollbackToBoundary(parser, currentConstrainedOut, "FROM");
            var repairedJoin := helpers.RollbackToBoundary(parser, currentConstrainedOut, "JOIN");
            var repairedOn := helpers.RollbackToBoundary(parser, currentConstrainedOut, "ON");

            // Pick the longest rollback that actually shortens the prefix
            var repaired := currentConstrainedOut;
            var bestLen := |currentConstrainedOut|;

            if |repairedComma| < bestLen {
              repaired := repairedComma;
              bestLen := |repairedComma|;
            }
            if |repairedWhere| < bestLen && |repairedWhere| > |repairedComma| {
              repaired := repairedWhere;
              bestLen := |repairedWhere|;
            }
            if |repairedFrom| < bestLen && |repairedFrom| > |repairedComma| {
              repaired := repairedFrom;
              bestLen := |repairedFrom|;
            }
            if |repairedJoin| < bestLen && |repairedJoin| > |repairedComma| {
              repaired := repairedJoin;
              bestLen := |repairedJoin|;
            }
            if |repairedOn| < bestLen && |repairedOn| > |repairedComma| {
              repaired := repairedOn;
              bestLen := |repairedOn|;
            }

            // If no boundary found, fall back to FROM
            if bestLen == |currentConstrainedOut| {
              repaired := repairedFrom;
            }

            generated := generated[..|generated| - (|currentConstrainedOut| - |repaired|)];
            currentConstrainedOut := repaired;
            insideConstrainedOut := true;
            steps := steps + 1;
          } else {
            var stablePrefix := generated[..|generated| - |currentConstrainedOut|];
            var constrainedPrompt := prompt + stablePrefix;
            var next := helpers.GroupBoostedConstrainedStep(lm, parser, constrainedPrompt, currentConstrainedOut, validTokenGroups, 1.5, eosToken);
            steps := steps + 1;
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
    }

    cost := steps;
    if maxSteps > 0 && cost == 0 { cost := 1; }
  }
}
