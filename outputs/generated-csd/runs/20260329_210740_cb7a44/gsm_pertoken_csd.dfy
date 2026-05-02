include "VerifiedAgentSynthesis.dfy"

module GeneratedCSD {
  import opened VerifiedDecoderAgent

  method MyCSDStrategy(lm: LM, parser: Parser, prompt: Prefix, maxSteps: nat, eosToken: Token) returns (generated: Prefix, cost: int)
    modifies lm.Logits
    requires lm.ValidTokensIdsLogits()
    requires parser.IsValidPrefix([])
    requires forall t: Token :: t in parser.ValidNextTokens([]) ==> t in lm.Tokens
    requires "<<" in lm.Tokens && ">>" in lm.Tokens
    ensures lm.ValidTokensIdsLogits()
    ensures |generated| <= maxSteps
    ensures cost <= maxSteps

  {
    var helpers := new CSDHelpers();
    // CSD_RATIONALE_BEGIN
    // Unconstrained phase: penalize << to force reasoning first.
    // Constrained phase adapts by expression length:
    //   < 5 tokens: penalize >> to prevent premature closing.
    //   >= 10 tokens: boost >> to encourage closure on long expressions.
    //   5-9 tokens: plain ConstrainedStep.
    // CSD_RATIONALE_END

    generated := [];
    var steps := 0;
    var insideConstrained := false;
    var currentConstrained: Prefix := [];

    while steps < maxSteps
      invariant 0 <= steps <= maxSteps
      invariant |generated| == steps
      invariant lm.ValidTokensIdsLogits()
      invariant insideConstrained ==> parser.IsValidPrefix(currentConstrained)
      invariant insideConstrained ==> forall t: Token ::
        t in parser.ValidNextTokens(currentConstrained) ==> t in lm.Tokens
      invariant helpers.cost == steps
      decreases maxSteps - steps, if insideConstrained then 1 else 0
    {
      if !insideConstrained {
        // UNCONSTRAINED PHASE — RAW pattern
        lm.GenerateLogits(prompt + generated);
        if steps < 15 {
          helpers.PenalizeTokenLogits(lm, ["<<"], 8.0);
        }
        var next := lm.ChooseNextTokenUnconstrained();
        helpers.cost := helpers.cost + 1;
        if next == eosToken { break; }
        generated := generated + [next];
        steps := steps + 1;
        if Contains(next, "<<") {
          insideConstrained := true;
          currentConstrained := [];
          CSDHelpers.RollbackPreservesTokenInvariant(lm, parser, []);
        }
      } else {
        // CONSTRAINED PHASE
        var narrow := helpers.DeadEndDetection(parser, currentConstrained, 3);
        if narrow || parser.IsCompletePrefix(currentConstrained) {
          insideConstrained := false;
        } else {
          var next;
          if |currentConstrained| < 5 {
            // Penalize >> when expression short — prevents premature closing
            next := helpers.PenalizedConstrainedStep(lm, parser, prompt, currentConstrained, [">>"], 3.0);
          } else if |currentConstrained| >= 10 {
            // Boost >> when expression long — prevents runaway number generation
            next := helpers.BoostedConstrainedStep(lm, parser, prompt, currentConstrained, [">>"], 50.0);
          } else {
            // Normal constrained step for mid-length expressions
            next := helpers.ConstrainedStep(lm, parser, prompt, currentConstrained);
          }
          generated := generated + [next];
          currentConstrained := currentConstrained + [next];
          steps := steps + 1;
          if Contains(next, ">>") {
            insideConstrained := false;
          }
        }
      }
    }
    cost := helpers.cost;
  }
}
