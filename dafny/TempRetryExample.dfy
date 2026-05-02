include "VerifiedAgentSynthesis.dfy"

module GeneratedCSD {
  import opened VerifiedDecoderAgent

  method MyCSDStrategy(lm: LM, parser: Parser, prompt: Prefix, currentPrefix: Prefix, maxSteps: nat, eosToken: Token) returns (generated: Prefix, cost: int)
    modifies lm.Logits
    requires lm.ValidTokensIdsLogits()
    requires parser.IsValidPrefix(currentPrefix)
    requires "<<" in lm.Tokens && ">>" in lm.Tokens
    requires eosToken in lm.Tokens
    ensures lm.ValidTokensIdsLogits()
    ensures parser.IsValidPrefix(generated)
    ensures exists suffix: Prefix :: generated == currentPrefix + suffix
    ensures |generated| <= |currentPrefix| + maxSteps
    ensures cost <= maxSteps
  {
    var helpers := new CSDHelpers();
    generated := currentPrefix;
    helpers.cost := 0;
    cost := 0;
    var suffix: Prefix := [];
    var attempts: nat := 0;

    while attempts < maxSteps
      invariant 0 <= attempts <= maxSteps
      invariant lm.ValidTokensIdsLogits()
      invariant parser.IsValidPrefix(generated)
      invariant generated == currentPrefix + suffix
      invariant |suffix| <= attempts
      invariant helpers.cost <= attempts
      decreases maxSteps - attempts
    {
      if parser.IsCompletePrefix(generated) {
        break;
      } else {
        var next, isValid := helpers.SoftConstrainedStep(lm, parser, prompt, generated, 2.0, eosToken);
        if next == eosToken {
          break;
        } else {
          attempts := attempts + 1;
          if isValid {
            generated := generated + [next];
            suffix := suffix + [next];
          }
        }
      }
    }
    cost := helpers.cost;
  }
}
