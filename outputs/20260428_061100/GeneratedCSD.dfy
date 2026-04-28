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
    var reasoning_steps := 0;
    var answer_steps := 0;
    var min_reasoning_steps := 1;
    var min_answer_steps := 2;
    var close_after_complete := 0;
    while ((stepsLeft > 0) && (phase < 4))
      invariant helpers.lm == lm
      invariant helpers.parser == parser
      invariant lm.ValidTokensIdsLogits()
      invariant 0 <= stepsLeft <= maxSteps
      invariant |generated| + stepsLeft <= maxSteps
      decreases stepsLeft
    {
      if ((phase == 0) && (reasoning_steps < min_reasoning_steps) && (stepsLeft > 0)) {
        generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
        reasoning_steps := (reasoning_steps + 1);
        if reasoning_steps >= min_reasoning_steps {
          phase := 1;
        }
      } else {
        if ((phase == 1) && (stepsLeft > 0)) {
          generated, stepsLeft := helpers.AppendLeftDelimiter(generated, stepsLeft);
          phase := 2;
        } else {
          if ((phase == 2) && (helpers.CanConstrain(generated))) {
            generated, stepsLeft := helpers.AppendConstrainedStep(prompt, generated, stepsLeft);
            answer_steps := (answer_steps + 1);
            if answer_steps >= min_answer_steps {
              close_after_complete := 1;
            }
          } else {
            if ((phase == 2) && (parser.IsCompletePrefix(helpers.LongestValidSuffix(generated))) && (helpers.CanExtendConstrained(generated)) && (answer_steps < min_answer_steps)) {
              generated, stepsLeft := helpers.AppendExtendConstrainedStep(prompt, generated, stepsLeft);
              answer_steps := (answer_steps + 1);
              if answer_steps >= min_answer_steps {
                close_after_complete := 1;
              }
            } else {
              if ((phase == 2) && (parser.IsCompletePrefix(helpers.LongestValidSuffix(generated))) && (close_after_complete > 0)) {
                generated, stepsLeft := helpers.AppendRightDelimiter(generated, stepsLeft);
                phase := 3;
              } else {
                if ((phase == 2) && (parser.IsCompletePrefix(helpers.LongestValidSuffix(generated))) && (!helpers.CanExtendConstrained(generated))) {
                  generated, stepsLeft := helpers.AppendRightDelimiter(generated, stepsLeft);
                  phase := 3;
                } else {
                  break;
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