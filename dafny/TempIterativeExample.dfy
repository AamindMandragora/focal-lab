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
    var steps: nat := 0;

    while steps < maxSteps
      invariant 0 <= steps <= maxSteps
      invariant lm.ValidTokensIdsLogits()
      invariant parser.IsValidPrefix(generated)
      invariant generated == currentPrefix + suffix
      invariant |suffix| == steps
      invariant helpers.cost <= steps
      decreases maxSteps - steps
    {
      if parser.IsCompletePrefix(generated) {
        break;
      } else {
        var validCount := helpers.ValidTokenCount(parser, generated);
        if validCount <= 4 {
          var next := helpers.ConstrainedStep(lm, parser, prompt, generated, eosToken);
          if next == eosToken {
            break;
          } else {
            generated := generated + [next];
            suffix := suffix + [next];
            steps := steps + 1;
          }
        } else {
          var next, isValid := helpers.SoftConstrainedStep(lm, parser, prompt, generated, 3.0, eosToken);
          if next == eosToken {
            break;
          } else {
            if isValid {
              generated := generated + [next];
              suffix := suffix + [next];
              steps := steps + 1;
            } else {
              break;
            }
          }
        }
      }
    }
    cost := helpers.cost;
  }
}
