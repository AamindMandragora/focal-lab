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
    var closed_spans := 0;
    var freeform_steps := 0;
    var nudge_mode := false;
    while stepsLeft > 0
      invariant helpers.lm == lm
      invariant helpers.parser == parser
      invariant lm.ValidTokensIdsLogits()
      invariant 0 <= stepsLeft <= maxSteps
      invariant |generated| + stepsLeft <= maxSteps
      decreases stepsLeft
    {
      if closed_spans > 0 {
        break;
      } else {
        if helpers.EndsWithRightDelimiter(generated) {
          closed_spans := closed_spans + 1;
          break;
        } else {
          if helpers.EndsWithLeftDelimiter(generated) {
            if helpers.IsComplete(generated) {
              generated, stepsLeft := helpers.AppendConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft);
            } else {
              if helpers.CanConstrain(generated) {
                generated, stepsLeft := helpers.AppendConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft);
              } else {
                break;
              }
            }
          } else {
            var min_needed := helpers.MinStepsToComplete(generated);
            var distance := helpers.ParserDistanceToComplete(generated);
            var valid_count := helpers.ValidContinuationCount(generated);
            var needed := 3;
            if min_needed > needed {
              needed := min_needed;
            }
            if distance > needed {
              needed := distance;
            }
            if valid_count == 1 {
              needed := (needed + 1);
            }
            var budget_pressure := !helpers.HasBudget(stepsLeft, (needed + 2));
            var should_nudge := ((nudge_mode) || (freeform_steps >= 12) || (budget_pressure));
            if should_nudge {
              nudge_mode := true;
              generated, stepsLeft := helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft);
            } else {
              generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
              freeform_steps := (freeform_steps + 1);
            }
          }
        }
      }
    }
    remainingSteps := stepsLeft;
  }

}