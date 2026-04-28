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
    var closed_spans := 0;
    var reason_steps := 0;
    var final_ready := 0;
    var saw_transition := 0;
    var next_token := eosToken;
    var new_steps := stepsLeft;
    while ((stepsLeft > 0) && (phase < 3) && (closed_spans < 2))
      invariant helpers.lm == lm
      invariant helpers.parser == parser
      invariant lm.ValidTokensIdsLogits()
      invariant 0 <= stepsLeft <= maxSteps
      invariant |generated| + stepsLeft <= maxSteps
      decreases stepsLeft
    {
      if phase == 0 {
        if ((final_ready == 0) && (helpers.HasBudget(stepsLeft, 18))) {
          generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
          reason_steps := (reason_steps + 1);
          if reason_steps >= 12 {
            saw_transition := 1;
          }
          if ((saw_transition > 0) && (!helpers.HasBudget(stepsLeft, 12))) {
            final_ready := 1;
          }
        } else {
          next_token, new_steps := helpers.UnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft);
          generated := (generated + [next_token]);
          stepsLeft := new_steps;
          if ((next_token == LeftDelimiter) || (next_token == SpacedLeftDelimiter)) {
            phase := 1;
          } else {
            reason_steps := (reason_steps + 1);
            if ((saw_transition > 0) && (!helpers.HasBudget(stepsLeft, 10))) {
              final_ready := 1;
            }
          }
        }
      } else {
        if ((phase == 1) && (((helpers.CanConstrain(generated)) || (parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)))))) {
          next_token, new_steps := helpers.ConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft);
          generated := (generated + [next_token]);
          stepsLeft := new_steps;
          if ((next_token == RightDelimiter) || (next_token == SpacedRightDelimiter)) {
            closed_spans := (closed_spans + 1);
            phase := 3;
          } else {
            if parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)) {
              saw_transition := 1;
            }
          }
        } else {
          break;
        }
      }
    }
    remainingSteps := stepsLeft;
  }

}