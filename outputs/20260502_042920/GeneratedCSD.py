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
    # Strategy: keep a durable free-form reasoning phase with no tiny fixed token
    # quota. Use unconstrained decoding for ordinary GSM reasoning, and only switch
    # to delimiter nudging after substantial reasoning progress or when remaining
    # budget and parser-completion signals jointly suggest it is time to finalize.
    # Track span state explicitly with phase/inside_span/closed_spans rather than
    # relying on delimiter suffixes as persistent state. In natural-delimiter mode,
    # open the final answer span by repeatedly using
    # AppendUnconstrainedNudgeLeftDelimiterStep until a left delimiter appears.
    # Once inside the span, use the positive guard
    # helpers.IsComplete(generated) or helpers.CanConstrain(generated), then emit
    # with AppendConstrainedOrRightDelimiterStep until a right delimiter closes the
    # span. Close after one completed answer span, and avoid tiny fixed phase quotas
    # for reasoning or answer length.
    # CSD_RATIONALE_END
    phase = "reason"
    inside_span = False
    closed_spans = 0
    reasoning_steps = 0
    nudge_steps = 0

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
        elif phase == "done":
            break
        elif inside_span or phase == "span":
            if helpers.EndsWithRightDelimiter(generated):
                inside_span = False
                closed_spans = closed_spans + 1
                phase = "done"
                break
            elif helpers.IsComplete(generated) or helpers.CanConstrain(generated):
                generated, stepsLeft = helpers.AppendConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft)
                if helpers.EndsWithRightDelimiter(generated):
                    inside_span = False
                    closed_spans = closed_spans + 1
                    phase = "done"
                    break
                else:
                    inside_span = True
                    phase = "span"
            else:
                break
        elif helpers.EndsWithLeftDelimiter(generated):
            inside_span = True
            phase = "span"
            if helpers.IsComplete(generated) or helpers.CanConstrain(generated):
                generated, stepsLeft = helpers.AppendConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft)
                if helpers.EndsWithRightDelimiter(generated):
                    inside_span = False
                    closed_spans = closed_spans + 1
                    phase = "done"
                    break
                else:
                    inside_span = True
                    phase = "span"
            else:
                break
        elif phase == "reason":
            answer_ready = False

            if reasoning_steps >= 24:
                answer_ready = True
            elif reasoning_steps >= 16:
                if stepsLeft <= 20:
                    answer_ready = True
                elif helpers.ValidContinuationCount(generated) <= 2:
                    answer_ready = True
                elif helpers.ParserDistanceToComplete(generated) <= 4:
                    answer_ready = True
                elif helpers.MinStepsToComplete(generated) <= 4:
                    answer_ready = True

            if answer_ready:
                phase = "nudge"
                generated, stepsLeft = helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft)
                nudge_steps = nudge_steps + 1
                if helpers.EndsWithLeftDelimiter(generated):
                    inside_span = True
                    phase = "span"
            else:
                generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
                reasoning_steps = reasoning_steps + 1
        elif phase == "nudge":
            if helpers.EndsWithLeftDelimiter(generated):
                inside_span = True
                phase = "span"
                if helpers.IsComplete(generated) or helpers.CanConstrain(generated):
                    generated, stepsLeft = helpers.AppendConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft)
                    if helpers.EndsWithRightDelimiter(generated):
                        inside_span = False
                        closed_spans = closed_spans + 1
                        phase = "done"
                        break
                    else:
                        inside_span = True
                        phase = "span"
                else:
                    break
            elif stepsLeft <= 6:
                break
            else:
                generated, stepsLeft = helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft)
                nudge_steps = nudge_steps + 1
                if helpers.EndsWithLeftDelimiter(generated):
                    inside_span = True
                    phase = "span"
        else:
            break

        if stepsLeft >= stepsLeftBeforeIteration:
            break
    remainingSteps = stepsLeft
    return generated, remainingSteps
