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
    var reason_signal := 0;
    var final_ready := 0;
    var scratch_goal := 1;
    var close_ready := 0;
    var next_token := eosToken;
    var new_steps := stepsLeft;
    while ((stepsLeft > 0) && (phase < 3) && (closed_spans < 4))
      invariant helpers.lm == lm
      invariant helpers.parser == parser
      invariant lm.ValidTokensIdsLogits()
      invariant 0 <= stepsLeft <= maxSteps
      invariant |generated| + stepsLeft <= maxSteps
      decreases stepsLeft
    {
      if phase == 0 {
        if ((final_ready == 0) && (closed_spans == 0) && (helpers.HasBudget(stepsLeft, 18))) {
          generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
          reason_signal := (reason_signal + 1);
          if reason_signal >= 6 {
            final_ready := 1;
          }
        } else {
          if ((final_ready == 0) && (closed_spans > 0) && (helpers.HasBudget(stepsLeft, 12))) {
            generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
            reason_signal := (reason_signal + 1);
            if reason_signal >= 9 {
              final_ready := 1;
            }
          } else {
            next_token := eosToken;
            new_steps := stepsLeft;
            if helpers.HasBudget(stepsLeft, 10) {
              next_token, new_steps := helpers.UnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft);
            } else {
              next_token, new_steps := helpers.UnconstrainedAllowLeftDelimiterStep(prompt, generated, stepsLeft);
            }
            generated := (generated + [next_token]);
            stepsLeft := new_steps;
            if ((next_token == LeftDelimiter) || (next_token == " <<")) {
              phase := 1;
              close_ready := 0;
            } else {
              reason_signal := (reason_signal + 1);
              if closed_spans > 0 {
                final_ready := 1;
              }
            }
          }
        }
      } else {
        if ((phase == 1) && (parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)))) {
          if ((close_ready > 0) || (closed_spans >= scratch_goal) || (!helpers.CanExtendConstrained(generated)) || (!helpers.HasBudget(stepsLeft, 2))) {
            next_token, new_steps := helpers.ConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft);
            generated := (generated + [next_token]);
            stepsLeft := new_steps;
            if ((next_token == RightDelimiter) || (next_token == " >>")) {
              closed_spans := (closed_spans + 1);
              reason_signal := 0;
              close_ready := 0;
              if ((final_ready > 0) && (closed_spans > scratch_goal)) {
                phase := 2;
              } else {
                phase := 0;
                if closed_spans >= scratch_goal {
                  final_ready := 1;
                }
              }
            } else {
              close_ready := (close_ready + 1);
            }
          } else {
            if helpers.CanExtendConstrained(generated) {
              next_token, new_steps := helpers.ConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft);
              generated := (generated + [next_token]);
              stepsLeft := new_steps;
              if ((next_token == RightDelimiter) || (next_token == " >>")) {
                closed_spans := (closed_spans + 1);
                reason_signal := 0;
                close_ready := 0;
                if ((final_ready > 0) && (closed_spans > scratch_goal)) {
                  phase := 2;
                } else {
                  phase := 0;
                  if closed_spans >= scratch_goal {
                    final_ready := 1;
                  }
                }
              } else {
                close_ready := (close_ready + 1);
              }
            } else {
              break;
            }
          }
        } else {
          if ((phase == 1) && (helpers.CanConstrain(generated))) {
            next_token, new_steps := helpers.ConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft);
            generated := (generated + [next_token]);
            stepsLeft := new_steps;
            if ((next_token == RightDelimiter) || (next_token == " >>")) {
              closed_spans := (closed_spans + 1);
              reason_signal := 0;
              close_ready := 0;
              if ((final_ready > 0) && (closed_spans > scratch_goal)) {
                phase := 2;
              } else {
                phase := 0;
                if closed_spans >= scratch_goal {
                  final_ready := 1;
                }
              }
            } else {
              if parser.ParserDistanceToComplete(helpers.LongestValidSuffix(generated)) <= 1 {
                close_ready := (close_ready + 1);
              }
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