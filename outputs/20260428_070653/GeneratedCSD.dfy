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
    var phase := 0;
    var opened_span := 0;
    var extended_once := 0;
    var post_close_steps := 0;
    while ((stepsLeft > 0) && (phase < 4))
      invariant helpers.lm == lm
      invariant helpers.parser == parser
      invariant lm.ValidTokensIdsLogits()
      invariant 0 <= stepsLeft <= maxSteps
      invariant |generated| + stepsLeft <= maxSteps
      decreases stepsLeft
    {
      if ((phase == 0) && (opened_span == 0) && (|generated| == 0)) {
        generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
      } else {
        if ((phase == 0) && (opened_span == 0) && (stepsLeft > (helpers.MinStepsToComplete(generated) + 2))) {
          generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
          if |generated| > 0 {
            phase := 1;
          }
        } else {
          if ((phase == 0) && (opened_span == 0)) {
            generated, stepsLeft := helpers.AppendLeftDelimiter(generated, stepsLeft);
            opened_span := 1;
            phase := 2;
          } else {
            if ((phase == 1) && (opened_span == 0)) {
              generated, stepsLeft := helpers.AppendLeftDelimiter(generated, stepsLeft);
              opened_span := 1;
              phase := 2;
            } else {
              if ((phase == 2) && (helpers.CanConstrain(generated))) {
                generated, stepsLeft := helpers.AppendConstrainedStep(prompt, generated, stepsLeft);
              } else {
                if ((phase == 2) && (parser.IsCompletePrefix(helpers.LongestValidSuffix(generated))) && (helpers.CanExtendConstrained(generated)) && (extended_once == 0) && (stepsLeft > 1)) {
                  generated, stepsLeft := helpers.AppendExtendConstrainedStep(prompt, generated, stepsLeft);
                  extended_once := 1;
                } else {
                  if ((phase == 2) && (parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)))) {
                    var suffix := helpers.LongestValidSuffix(generated);
                    var distance := parser.ParserDistanceToComplete(suffix);
                    var continuations := parser.ValidContinuationCount(suffix);
                    if ((distance == 0) && (((continuations <= 1) || (stepsLeft <= 2) || (extended_once > 0)))) {
                      generated, stepsLeft := helpers.AppendRightDelimiter(generated, stepsLeft);
                      phase := 3;
                    } else {
                      if helpers.CanExtendConstrained(generated) {
                        generated, stepsLeft := helpers.AppendExtendConstrainedStep(prompt, generated, stepsLeft);
                        extended_once := 1;
                      } else {
                        generated, stepsLeft := helpers.AppendRightDelimiter(generated, stepsLeft);
                        phase := 3;
                      }
                    }
                  } else {
                    if ((phase == 3) && (post_close_steps == 0)) {
                      generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
                      post_close_steps := 1;
                    } else {
                      if phase == 3 {
                        break;
                      } else {
                        generated, stepsLeft := helpers.AppendBudgetAwareStep(prompt, generated, stepsLeft, 1);
                      }
                    }
                  }
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