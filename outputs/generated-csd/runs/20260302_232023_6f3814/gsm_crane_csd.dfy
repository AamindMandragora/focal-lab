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
// Use CraneGeneration to detect << delimiters and switch to constrained mode, closing at >>.
// This approach ensures that reasoning text is interspersed with math expressions, producing
// multi-step arithmetic solutions with one short expression per <<>> window.
// CSD_RATIONALE_END
generated := helpers.CraneGeneration(lm, parser, prompt, maxSteps, 5, eosToken);
cost := helpers.cost;
    cost := helpers.cost;
  }
}
