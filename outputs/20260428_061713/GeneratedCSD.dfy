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
    var reason_steps := 0;
    var answer_steps := 0;
    var min_reason := 8;
    var max_reason := 40;
    var close_ready := 0;
    while ((stepsLeft > 0) && (phase < 3))
      invariant helpers.lm == lm
      invariant helpers.parser == parser
      invariant lm.ValidTokensIdsLogits()
      invariant 0 <= stepsLeft <= maxSteps
      invariant |generated| + stepsLeft <= maxSteps
      decreases stepsLeft
    {
      if phase == 0 {
        if ((reason_steps < min_reason) && (stepsLeft > (helpers.MinStepsToComplete(generated) + 3))) {
          generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
          reason_steps := (reason_steps + 1);
        } else {
          if ((reason_steps < max_reason) && (stepsLeft > (helpers.MinStepsToComplete(generated) + 6))) {
            generated, stepsLeft := helpers.AppendBudgetAwareStep(prompt, generated, stepsLeft, 2);
            reason_steps := (reason_steps + 1);
          } else {
            generated, stepsLeft := helpers.AppendLeftDelimiter(generated, stepsLeft);
            phase := 1;
          }
        }
      } else {
        if ((phase == 1) && (helpers.CanConstrain(generated))) {
          if answer_steps == 0 {
            generated, stepsLeft := helpers.AppendTopKConstrainedStep(prompt, generated, 1, stepsLeft);
          } else {
            generated, stepsLeft := helpers.AppendConstrainedStep(prompt, generated, stepsLeft);
          }
          answer_steps := (answer_steps + 1);
        } else {
          if ((phase == 1) && (parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)))) {
            if ((helpers.CanExtendConstrained(generated)) && (answer_steps < 12) && (stepsLeft > 1) && (close_ready < 2)) {
              generated, stepsLeft := helpers.AppendExtendConstrainedStep(prompt, generated, stepsLeft);
              answer_steps := (answer_steps + 1);
              close_ready := (close_ready + 1);
            } else {
              generated, stepsLeft := helpers.AppendRightDelimiter(generated, stepsLeft);
              phase := 2;
            }
          } else {
            break;
          }
        }
      }
    }
    remainingSteps := stepsLeft;
  }

}