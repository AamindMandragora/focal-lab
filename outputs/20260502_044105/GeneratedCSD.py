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
    # Strategy: maintain explicit phase state for natural-delimiter decoding.
    # In the reasoning phase, emit ordinary unconstrained tokens until an
    # answer-ready signal is reached based on modest reasoning progress together
    # with parser proximity indicators. After that, switch to a nudge phase that
    # still decodes unconstrained text but nudges the LM to emit the left delimiter
    # naturally. Once a left delimiter is actually emitted, enter span mode.
    # Inside the span, use the positive guard
    # `helpers.IsComplete(generated) or helpers.CanConstrain(generated)` and then
    # call `helpers.AppendConstrainedOrRightDelimiterStep(...)` so the LM may emit
    # constrained answer tokens or naturally close with the right delimiter once
    # completion is available. A real `closed_spans` counter records closure, and
    # every loop branch either consumes one helper step or breaks.
    # CSD_RATIONALE_END
    phase = "reason"
    reasoning_steps = 0
    closed_spans = 0
    moderate_reasoning_threshold = 8
    late_reasoning_threshold = 14

    # invariant helpers.lm == lm
    # invariant helpers.parser == parser
    # invariant lm.ValidTokensIdsLogits()
    # invariant 0 <= stepsLeft <= maxSteps
    # invariant |generated| + stepsLeft <= maxSteps
    # decreases stepsLeft
    while stepsLeft > 0:
        stepsLeftBeforeIteration = stepsLeft
        if closed_spans > 0:
            break
        elif phase == "reason":
            if helpers.EndsWithRightDelimiter(generated):
                closed_spans = closed_spans + 1
                break
            elif helpers.EndsWithLeftDelimiter(generated):
                phase = "inside_span"
                if helpers.IsComplete(generated) or helpers.CanConstrain(generated):
                    generated, stepsLeft = helpers.AppendConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft)
                    if helpers.EndsWithRightDelimiter(generated):
                        closed_spans = closed_spans + 1
                        break
                else:
                    break
            else:
                answer_ready = False
                if reasoning_steps >= late_reasoning_threshold:
                    answer_ready = True
                elif reasoning_steps >= moderate_reasoning_threshold:
                    if helpers.IsComplete(generated):
                        answer_ready = True
                    elif (helpers.IsComplete(generated) or helpers.CanConstrain(generated)):
                        answer_ready = True
                    elif helpers.ValidContinuationCount(generated) <= 1:
                        answer_ready = True
                    elif helpers.ParserDistanceToComplete(generated) <= 1:
                        answer_ready = True
                    elif helpers.MinStepsToComplete(generated) <= 1:
                        answer_ready = True
                if answer_ready:
                    phase = "nudge_left"
                    generated, stepsLeft = helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft)
                    if helpers.EndsWithRightDelimiter(generated):
                        closed_spans = closed_spans + 1
                        break
                    elif helpers.EndsWithLeftDelimiter(generated):
                        phase = "inside_span"
                    else:
                        phase = "nudge_left"
                else:
                    generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
                    reasoning_steps = reasoning_steps + 1
                    if helpers.EndsWithRightDelimiter(generated):
                        closed_spans = closed_spans + 1
                        break
                    elif helpers.EndsWithLeftDelimiter(generated):
                        phase = "inside_span"
                    else:
                        phase = "reason"
        elif phase == "nudge_left":
            if helpers.EndsWithRightDelimiter(generated):
                closed_spans = closed_spans + 1
                break
            elif helpers.EndsWithLeftDelimiter(generated):
                phase = "inside_span"
                if helpers.IsComplete(generated) or helpers.CanConstrain(generated):
                    generated, stepsLeft = helpers.AppendConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft)
                    if helpers.EndsWithRightDelimiter(generated):
                        closed_spans = closed_spans + 1
                        break
                else:
                    break
            else:
                generated, stepsLeft = helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft)
                if helpers.EndsWithRightDelimiter(generated):
                    closed_spans = closed_spans + 1
                    break
                elif helpers.EndsWithLeftDelimiter(generated):
                    phase = "inside_span"
                else:
                    phase = "nudge_left"
        elif phase == "inside_span":
            if helpers.EndsWithRightDelimiter(generated):
                closed_spans = closed_spans + 1
                break
            elif helpers.IsComplete(generated) or helpers.CanConstrain(generated):
                generated, stepsLeft = helpers.AppendConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft)
                if helpers.EndsWithRightDelimiter(generated):
                    closed_spans = closed_spans + 1
                    break
                else:
                    phase = "inside_span"
            else:
                break
        else:
            generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
            if helpers.EndsWithRightDelimiter(generated):
                closed_spans = closed_spans + 1
                break
            elif helpers.EndsWithLeftDelimiter(generated):
                phase = "inside_span"
            else:
                phase = "reason"
        if stepsLeft >= stepsLeftBeforeIteration:
            break
    remainingSteps = stepsLeft
    return generated, remainingSteps
