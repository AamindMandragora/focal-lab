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
    var inside_span := false;
    var closed_spans := 0;
    var nudge_mode := false;
    var final_ready := false;
    var scratch_mode := false;
    var scratch_ready := false;
    var opening_scratch_span := false;
    var current_span_is_final := false;
    while stepsLeft > 0
      invariant helpers.lm == lm
      invariant helpers.parser == parser
      invariant lm.ValidTokensIdsLogits()
      invariant 0 <= stepsLeft <= maxSteps
      invariant |generated| + stepsLeft <= maxSteps
      decreases stepsLeft
    {
      var stepsLeftBeforeIteration := stepsLeft;
      if inside_span {
        if helpers.EndsWithRightDelimiter(generated) {
          inside_span := false;
          phase := "post_span";
          closed_spans := (closed_spans + 1);
          if opening_scratch_span {
            scratch_mode := false;
            scratch_ready := false;
            opening_scratch_span := false;
          }
          if ((current_span_is_final) && (final_ready)) {
            break;
          }
          current_span_is_final := false;
          generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
        } else {
          if ((helpers.IsComplete(generated)) || (helpers.CanConstrain(generated))) {
            generated, stepsLeft := helpers.AppendConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft);
            if helpers.EndsWithRightDelimiter(generated) {
              inside_span := false;
              phase := "post_span";
              closed_spans := (closed_spans + 1);
              if opening_scratch_span {
                scratch_mode := false;
                scratch_ready := false;
                opening_scratch_span := false;
              }
              if ((current_span_is_final) && (final_ready)) {
                break;
              }
              current_span_is_final := false;
            }
          } else {
            break;
          }
        }
      } else {
        if helpers.EndsWithLeftDelimiter(generated) {
          inside_span := true;
          phase := "span";
          current_span_is_final := ((final_ready) && (!scratch_ready));
          if opening_scratch_span {
            scratch_mode := true;
            scratch_ready := false;
            current_span_is_final := false;
          }
          generated, stepsLeft := helpers.AppendConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft);
          if helpers.EndsWithRightDelimiter(generated) {
            inside_span := false;
            phase := "post_span";
            closed_spans := (closed_spans + 1);
            if opening_scratch_span {
              scratch_mode := false;
              scratch_ready := false;
              opening_scratch_span := false;
            }
            if ((current_span_is_final) && (final_ready)) {
              break;
            }
            current_span_is_final := false;
          }
        } else {
          if ((closed_spans >= 1) && (helpers.MinStepsToComplete(generated) <= 4)) {
            final_ready := true;
          } else {
            if ((closed_spans >= 2) && (helpers.ParserDistanceToComplete(generated) <= 3)) {
              final_ready := true;
            } else {
              if ((closed_spans >= 1) && (helpers.ValidContinuationCount(generated) <= 2)) {
                final_ready := true;
              }
            }
          }
          if final_ready {
            scratch_ready := false;
            scratch_mode := false;
            opening_scratch_span := false;
            nudge_mode := true;
            generated, stepsLeft := helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft);
            if helpers.EndsWithLeftDelimiter(generated) {
              inside_span := true;
              phase := "span";
              nudge_mode := false;
              opening_scratch_span := false;
              current_span_is_final := true;
            }
          } else {
            if nudge_mode {
              generated, stepsLeft := helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft);
              if helpers.EndsWithLeftDelimiter(generated) {
                inside_span := true;
                phase := "span";
                nudge_mode := false;
                opening_scratch_span := scratch_ready;
                current_span_is_final := false;
                if opening_scratch_span {
                  scratch_mode := true;
                  scratch_ready := false;
                }
              }
            } else {
              generated, stepsLeft := helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft);
              if ((closed_spans == 0) && (helpers.ParserDistanceToComplete(generated) <= 6)) {
                nudge_mode := true;
                scratch_ready := true;
                opening_scratch_span := false;
              } else {
                if ((closed_spans >= 1) && (helpers.ValidContinuationCount(generated) <= 4) && (!final_ready)) {
                  nudge_mode := true;
                  scratch_ready := true;
                  opening_scratch_span := false;
                } else {
                  if ((closed_spans >= 1) && (helpers.MinStepsToComplete(generated) <= 5) && (!final_ready)) {
                    nudge_mode := true;
                    scratch_ready := true;
                    opening_scratch_span := false;
                  }
                }
              }
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