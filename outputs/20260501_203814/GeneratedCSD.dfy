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
    var saw_closed_span := false;
    var final_mode := false;
    while stepsLeft > 0
      invariant helpers.lm == lm
      invariant helpers.parser == parser
      invariant lm.ValidTokensIdsLogits()
      invariant 0 <= stepsLeft <= maxSteps
      invariant |generated| + stepsLeft <= maxSteps
      decreases stepsLeft
    {
      if inside_span {
        if helpers.EndsWithRightDelimiter(generated) {
          inside_span := false;
          saw_closed_span := true;
          if final_mode {
            break;
          }
          generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
          if helpers.EndsWithLeftDelimiter(generated) {
            inside_span := true;
          } else {
            if helpers.EndsWithRightDelimiter(generated) {
              saw_closed_span := true;
            }
          }
          continue;
        }
        if helpers.IsComplete(generated) {
          generated, stepsLeft := helpers.AppendRightDelimiter(generated, stepsLeft);
          inside_span := false;
          saw_closed_span := true;
          if final_mode {
            break;
          }
          continue;
        }
        if helpers.CanConstrain(generated) {
          generated, stepsLeft := helpers.AppendConstrainedStep(prompt, generated, stepsLeft);
          if helpers.EndsWithRightDelimiter(generated) {
            inside_span := false;
            saw_closed_span := true;
          }
          continue;
        }
        break;
      } else {
        if helpers.EndsWithLeftDelimiter(generated) {
          inside_span := true;
          if helpers.IsComplete(generated) {
            generated, stepsLeft := helpers.AppendRightDelimiter(generated, stepsLeft);
            inside_span := false;
            saw_closed_span := true;
            if final_mode {
              break;
            }
          } else {
            if helpers.CanConstrain(generated) {
              generated, stepsLeft := helpers.AppendConstrainedStep(prompt, generated, stepsLeft);
              if helpers.EndsWithRightDelimiter(generated) {
                inside_span := false;
                saw_closed_span := true;
              }
            } else {
              break;
            }
          }
          continue;
        }
        if helpers.EndsWithRightDelimiter(generated) {
          saw_closed_span := true;
          if final_mode {
            break;
          }
          generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
          if helpers.EndsWithLeftDelimiter(generated) {
            inside_span := true;
          } else {
            if helpers.EndsWithRightDelimiter(generated) {
              saw_closed_span := true;
            }
          }
          continue;
        }
        if ((!final_mode) && (saw_closed_span)) {
          final_mode := true;
          generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
          if helpers.EndsWithLeftDelimiter(generated) {
            inside_span := true;
          } else {
            if helpers.EndsWithRightDelimiter(generated) {
              saw_closed_span := true;
            }
          }
          continue;
        }
        generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
        if helpers.EndsWithLeftDelimiter(generated) {
          inside_span := true;
        } else {
          if helpers.EndsWithRightDelimiter(generated) {
            saw_closed_span := true;
          }
        }
        continue;
      }
    }
    remainingSteps := stepsLeft;
  }

}