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
    var reasoning_seen := 0;
    var answer_tokens := 0;
    var completed_once := 0;
    var post_span_tokens := 0;
    while ((stepsLeft > 0) && (phase < 4))
      invariant helpers.lm == lm
      invariant helpers.parser == parser
      invariant lm.ValidTokensIdsLogits()
      invariant 0 <= stepsLeft <= maxSteps
      invariant |generated| + stepsLeft <= maxSteps
      decreases stepsLeft
    {
      if phase == 0 {
        if reasoning_seen == 0 {
          generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
          reasoning_seen := (reasoning_seen + 1);
        } else {
          if helpers.HasBudget(stepsLeft, (helpers.MinStepsToComplete(generated) + 2)) {
            generated, stepsLeft := helpers.AppendLeftDelimiter(generated, stepsLeft);
            phase := 1;
          } else {
            generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
            reasoning_seen := (reasoning_seen + 1);
          }
        }
      } else {
        if phase == 1 {
          if helpers.CanConstrain(generated) {
            generated, stepsLeft := helpers.AppendConstrainedStep(prompt, generated, stepsLeft);
            answer_tokens := (answer_tokens + 1);
            phase := 2;
          } else {
            break;
          }
        } else {
          if ((phase == 2) && (helpers.CanConstrain(generated))) {
            generated, stepsLeft := helpers.AppendConstrainedStep(prompt, generated, stepsLeft);
            answer_tokens := (answer_tokens + 1);
          } else {
            if ((phase == 2) && (parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)))) {
              if ((helpers.CanExtendConstrained(generated)) && (answer_tokens < 3) && (helpers.HasBudget(stepsLeft, 2))) {
                generated, stepsLeft := helpers.AppendExtendConstrainedStep(prompt, generated, stepsLeft);
                answer_tokens := (answer_tokens + 1);
                completed_once := 1;
              } else {
                if ((helpers.CanExtendConstrained(generated)) && (parser.ValidContinuationCount(helpers.LongestValidSuffix(generated)) > 1) && (helpers.HasBudget(stepsLeft, 2)) && (completed_once == 0)) {
                  generated, stepsLeft := helpers.AppendExtendConstrainedStep(prompt, generated, stepsLeft);
                  answer_tokens := (answer_tokens + 1);
                  completed_once := 1;
                } else {
                  generated, stepsLeft := helpers.AppendRightDelimiter(generated, stepsLeft);
                  phase := 3;
                }
              }
            } else {
              if phase == 3 {
                if ((stepsLeft > 1) && (post_span_tokens == 0)) {
                  generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
                  post_span_tokens := (post_span_tokens + 1);
                } else {
                  phase := 4;
                  break;
                }
              } else {
                break;
              }
            }
          }
        }
      }
    }
    remainingSteps := stepsLeft;
  }

}