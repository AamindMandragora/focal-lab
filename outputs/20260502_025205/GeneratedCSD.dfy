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
    var answer_pressure := false;
    while stepsLeft > 0
      invariant helpers.lm == lm
      invariant helpers.parser == parser
      invariant lm.ValidTokensIdsLogits()
      invariant 0 <= stepsLeft <= maxSteps
      invariant |generated| + stepsLeft <= maxSteps
      decreases stepsLeft
    {
      var stepsLeftBeforeIteration := stepsLeft;
      if in_span {
        if helpers.EndsWithRightDelimiter(generated) {
          in_span := false;
          closed_spans := (closed_spans + 1);
          if phase == "finalizing" {
            break;
          }
          phase := "reason";
          generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
        } else {
          if helpers.IsDead(generated) {
            break;
          } else {
            if ((helpers.IsComplete(generated)) || (helpers.CanConstrain(generated))) {
              generated, stepsLeft := helpers.AppendConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft);
              if helpers.EndsWithRightDelimiter(generated) {
                in_span := false;
                closed_spans := (closed_spans + 1);
                if phase == "finalizing" {
                  break;
                }
                phase := "reason";
              }
            } else {
              break;
            }
          }
        }
      } else {
        if helpers.EndsWithLeftDelimiter(generated) {
          in_span := true;
          phase := "finalizing";
          if ((helpers.IsComplete(generated)) || (helpers.CanConstrain(generated))) {
            generated, stepsLeft := helpers.AppendConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft);
            if helpers.EndsWithRightDelimiter(generated) {
              in_span := false;
              closed_spans := (closed_spans + 1);
              break;
            }
          } else {
            break;
          }
        } else {
          if phase == "reason" {
            if closed_spans == 0 {
              if !helpers.HasBudget(stepsLeft, 6) {
                answer_pressure := true;
              } else {
                if helpers.HasBudget(stepsLeft, 12) {
                  answer_pressure := false;
                }
              }
            } else {
              if !helpers.HasBudget(stepsLeft, 4) {
                answer_pressure := true;
              }
            }
            if answer_pressure {
              phase := "finalizing";
              generated, stepsLeft := helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft);
              if helpers.EndsWithLeftDelimiter(generated) {
                in_span := true;
              }
            } else {
              generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
              if helpers.EndsWithLeftDelimiter(generated) {
                in_span := true;
                phase := "finalizing";
              }
            }
          } else {
            generated, stepsLeft := helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft);
            if helpers.EndsWithLeftDelimiter(generated) {
              in_span := true;
            }
          }
        }
      }
      if stepsLeft >= stepsLeftBeforeIteration {
        break;
      }
    }
    remainingSteps := stepsLeft;
  }

}