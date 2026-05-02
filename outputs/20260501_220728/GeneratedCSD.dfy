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
    var phase := "reason";
    var in_span := false;
    var closed_spans := 0;
    while stepsLeft > 0
      invariant helpers.lm == lm
      invariant helpers.parser == parser
      invariant lm.ValidTokensIdsLogits()
      invariant 0 <= stepsLeft <= maxSteps
      invariant |generated| + stepsLeft <= maxSteps
      decreases stepsLeft
    {
      if in_span {
        if helpers.IsComplete(generated) {
          generated, stepsLeft := helpers.AppendRightDelimiter(generated, stepsLeft);
          in_span := false;
          closed_spans := (closed_spans + 1);
          phase := "done";
        } else {
          if helpers.CanConstrain(generated) {
            generated, stepsLeft := helpers.AppendConstrainedStep(prompt, generated, stepsLeft);
            if helpers.EndsWithRightDelimiter(generated) {
              in_span := false;
              closed_spans := (closed_spans + 1);
              phase := "done";
            }
          } else {
            break;
          }
        }
      } else {
        if phase == "done" {
          break;
        } else {
          if closed_spans > 0 {
            break;
          } else {
            if stepsLeft <= 2 {
              generated, stepsLeft := helpers.AppendLeftDelimiter(generated, stepsLeft);
              in_span := true;
              phase := "finalize";
            } else {
              generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
              if helpers.EndsWithLeftDelimiter(generated) {
                in_span := true;
                phase := "finalize";
              } else {
                if helpers.EndsWithRightDelimiter(generated) {
                  closed_spans := (closed_spans + 1);
                  phase := "done";
                }
              }
            }
          }
        }
      }
    }
    remainingSteps := stepsLeft;
  }

}