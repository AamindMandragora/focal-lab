include "VerifiedAgentSynthesis.dfy"

module GeneratedCSD {
  import opened VerifiedDecoderAgent

  method MyCSDStrategy(lm: LM, parser: Parser, prompt: Prefix, maxSteps: nat, eosToken: Token) returns (generated: Prefix, remainingSteps: nat)
    modifies lm.Logits
    requires lm.ValidTokensIdsLogits()
    requires parser.IsValidPrefix([])
    requires !parser.IsCompletePrefix([])
    requires forall t: Token :: t in parser.ValidNextTokens([]) ==> t in lm.Tokens
    requires maxSteps >= 2
    requires LeftDelimiter in lm.Tokens
    requires RightDelimiter in lm.Tokens
    ensures lm.ValidTokensIdsLogits()
    ensures |generated| <= maxSteps
    ensures remainingSteps >= 0 && remainingSteps <= maxSteps
  {
    var helpers := new CSDHelpers(lm, parser);
    lm.ValidTokensIdsLogitsAlways();
    generated := [];
    var stepsLeft := maxSteps;
    var inside_span := false;
    var closed_span := false;
    while stepsLeft > 0
      invariant helpers.lm == lm
      invariant helpers.parser == parser
      invariant lm.ValidTokensIdsLogits()
      invariant 0 <= stepsLeft <= maxSteps
      invariant |generated| + stepsLeft <= maxSteps
      decreases stepsLeft
    {
      if closed_span {
        break;
      } else {
        if inside_span {
          if helpers.IsComplete(generated) {
            generated, stepsLeft := helpers.AppendRightDelimiter(generated, stepsLeft);
            inside_span := false;
            closed_span := true;
          } else {
            if helpers.CanConstrain(generated) {
              generated, stepsLeft := helpers.AppendConstrainedStep(prompt, generated, stepsLeft);
            } else {
              break;
            }
          }
        } else {
          generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
          if helpers.EndsWithLeftDelimiter(generated) {
            inside_span := true;
          }
        }
      }
    }
    remainingSteps := stepsLeft;
  }

}