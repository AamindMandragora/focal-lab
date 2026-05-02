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
    

  {
    var helpers := new CSDHelpers();
    // CSD_RATIONALE_BEGIN
// For GSM-Symbolic reasoning, we need to generate short symbolic mathematical expressions that adhere to a strict arithmetic grammar with variables and numeric constants. Given the critical rules, we will use the CRANE-style generation approach. This approach starts with free-form text, switches to constrained mode when the model generates '<<', and continues in constrained mode until the parser completes (e.g., expression + '>>'). This ensures that the generated expressions are both syntactically correct and semantically meaningful.
// CSD_RATIONALE_END
generated := helpers.CraneGeneration(lm, parser, prompt, maxSteps, 10, eosToken);
cost := helpers.cost;
    cost := helpers.cost;
  }
}
