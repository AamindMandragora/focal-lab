from __future__ import annotations

from VerifiedAgentSynthesis import (
    CSDHelpers,
    LM,
    LeftDelimiter,
    RightDelimiter,
    SpacedLeftDelimiter,
    SpacedRightDelimiter,
    Parser,
    Prefix,
    Token,
    dafny_spec,
)


DAFNY_INCLUDE = "VerifiedAgentSynthesis.dfy"
MODULE_NAME = "GeneratedCSD"
DAFNY_OPEN_IMPORT = "VerifiedDecoderAgent"


@dafny_spec(
    kind="method",
    modifies=("lm.Logits",),
    requires=(
        "lm.ValidTokensIdsLogits()",
        "parser.IsValidPrefix([])",
        "!parser.IsCompletePrefix([])",
        "forall t: Token :: t in parser.ValidNextTokens([]) ==> t in lm.Tokens",
        "maxSteps >= 2",
        "LeftDelimiter in lm.Tokens",
        "RightDelimiter in lm.Tokens",
    ),
    ensures=(
        "lm.ValidTokensIdsLogits()",
        "|generated| <= maxSteps",
        "remainingSteps >= 0 && remainingSteps <= maxSteps",
    ),
)
def MyCSDStrategy(
    lm: LM,
    parser: Parser,
    prompt: Prefix,
    maxSteps: int,
    eosToken: Token,
) -> tuple[Prefix, int]:
    helpers = CSDHelpers(lm, parser)
    lm.ValidTokensIdsLogitsAlways()
    generated = []
    stepsLeft = maxSteps
    # CSD_RATIONALE_BEGIN
    # Strategy: natural-delimiter GSM symbolic reasoning.
    # - Use unconstrained decoding for ordinary prose and for the transition into a
    #   delimited answer span.
    # - Maintain explicit durable state for whether we are inside a verified span,
    #   whether we are in the final-answer phase, and how many spans have closed.
    # - Inside a span, use the positive guard
    #   (helpers.CanConstrain(generated)) before
    #   AppendConstrainedStep so complete expressions may close with >>.
    # - Delay the final span until an answer cue or budget pressure, preferring one
    #   compact final arithmetic expression/equation. If earlier scratch spans arise
    #   naturally they are allowed, but this policy biases toward a delayed final
    #   span for GSM.
    # - Use a checkpoint only for bounded local recovery from dead states.
    # CSD_RATIONALE_END
    phase = "reason"
    inside_span = False
    closed_spans = 0
    final_phase = False
    has_checkpoint = False
    checkpoint = []
    stall_count = 0

    # invariant helpers.lm == lm
    # invariant helpers.parser == parser
    # invariant lm.ValidTokensIdsLogits()
    # invariant 0 <= stepsLeft <= maxSteps
    # invariant |generated| + stepsLeft <= maxSteps
    # decreases stepsLeft
    while stepsLeft > 0:
        stepsLeftBeforeIteration = stepsLeft
        if helpers.IsDead(generated):
            if has_checkpoint:
                generated = helpers.RestoreCheckpoint(checkpoint)
                has_checkpoint = False
                generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
                if helpers.EndsWithLeftDelimiter(generated):
                    inside_span = True
                    phase = "span"
                elif helpers.EndsWithRightDelimiter(generated):
                    inside_span = False
                    closed_spans = closed_spans + 1
                    phase = "reason"
                else:
                    phase = "reason" if not inside_span else "span"
                    break
                break
            break

        if inside_span:
            if helpers.EndsWithRightDelimiter(generated):
                inside_span = False
                closed_spans = closed_spans + 1
                phase = "after_span"
                if final_phase:
                    break
                generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
                if helpers.EndsWithLeftDelimiter(generated):
                    inside_span = True
                    phase = "span"
                elif helpers.EndsWithRightDelimiter(generated):
                    inside_span = False
                    closed_spans = closed_spans + 1
                    phase = "after_span"
                else:
                    phase = "reason"
                    break
                break
            if helpers.CanConstrain(generated):
                if not has_checkpoint:
                    checkpoint = helpers.Checkpoint(generated)
                    has_checkpoint = True
                generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
                if helpers.EndsWithRightDelimiter(generated):
                    inside_span = False
                    closed_spans = closed_spans + 1
                    phase = "after_span"
                    if final_phase:
                        break
                else:
                    phase = "span"
                    break
                break
            break

        if phase == "after_span":
            if final_phase and closed_spans > 0:
                break
            generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
            if helpers.EndsWithLeftDelimiter(generated):
                inside_span = True
                phase = "span"
                has_checkpoint = False
            elif helpers.EndsWithRightDelimiter(generated):
                inside_span = False
                closed_spans = closed_spans + 1
                phase = "after_span"
            else:
                phase = "reason"
                break
            break
        answer_pressure = False
        if not helpers.HasBudget(stepsLeft, 6):
            answer_pressure = True
        elif closed_spans == 0 and helpers.HasBudget(stepsLeft, 1) and helpers.MinStepsToComplete(generated) <= 1:
            answer_pressure = True
        elif stall_count >= 4:
            answer_pressure = True

        if answer_pressure:
            final_phase = True
            phase = "seek_span"
        elif phase == "seek_span":
            final_phase = True

        if phase == "seek_span":
            generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
            if helpers.EndsWithLeftDelimiter(generated):
                inside_span = True
                phase = "span"
                has_checkpoint = False
            elif helpers.EndsWithRightDelimiter(generated):
                inside_span = False
                closed_spans = closed_spans + 1
                phase = "after_span"
            else:
                phase = "seek_span"
                break
            stall_count = stall_count + 1
            break
        generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
        if helpers.EndsWithLeftDelimiter(generated):
            inside_span = True
            phase = "span"
            has_checkpoint = False
            stall_count = 0
        elif helpers.EndsWithRightDelimiter(generated):
            inside_span = False
            closed_spans = closed_spans + 1
            phase = "after_span"
            stall_count = 0
        else:
            phase = "reason"
            stall_count = stall_count + 1
            break
        if stepsLeft >= stepsLeftBeforeIteration:
            break
        else:
            break
    remainingSteps = stepsLeft
    return generated, remainingSteps
