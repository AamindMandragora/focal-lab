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
    # Strategy: use a three-stage natural-delimiter policy tuned for GSM.
    # First, allow a substantial unconstrained reasoning runway so the model can set
    # up the full arithmetic chain instead of opening on an early intermediate
    # calculation. Second, enter an answer-seeking phase with repeated natural
    # left-delimiter nudges early enough that several attempts remain available.
    # Third, once a span is open, keep explicit span state and repeatedly use the
    # positive guard `helpers.IsComplete(generated) or helpers.CanConstrain(generated)`
    # before `AppendConstrainedOrRightDelimiterStep`, allowing the model to either
    # finish the final answer token sequence or close with `>>`. If the span is not
    # yet grammar-ready, continue ordinary unconstrained generation rather than
    # exiting. Stop after the first closed span.
    # CSD_RATIONALE_END
    phase = "reason"
    inside_span = False
    closed_spans = 0
    reason_steps = 0
    open_attempts = 0
    answer_seek_steps = 0
    setup_ready = False
    late_ready = False

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
        elif inside_span:
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
                    phase = "span"
            else:
                generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
                if helpers.EndsWithRightDelimiter(generated):
                    inside_span = False
                    closed_spans = closed_spans + 1
                    phase = "done"
                    break
                elif helpers.EndsWithLeftDelimiter(generated):
                    inside_span = True
                    phase = "span"
                else:
                    phase = "span"
        elif phase == "open":
            should_nudge = False
            if open_attempts < 3:
                should_nudge = True
            elif answer_seek_steps < 6:
                should_nudge = True
            elif late_ready and open_attempts < 8:
                should_nudge = True
            elif setup_ready and helpers.IsComplete(generated):
                should_nudge = True
            elif setup_ready and helpers.MinStepsToComplete(generated) <= 2:
                should_nudge = True
            elif setup_ready and helpers.ParserDistanceToComplete(generated) <= 2:
                should_nudge = True
            elif setup_ready and helpers.ValidContinuationCount(generated) <= 3:
                should_nudge = True

            if should_nudge:
                generated, stepsLeft = helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft)
                open_attempts = open_attempts + 1
                answer_seek_steps = answer_seek_steps + 1
                if helpers.EndsWithLeftDelimiter(generated):
                    inside_span = True
                    phase = "span"
                    open_attempts = 0
                else:
                    phase = "open"
            else:
                generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
                reason_steps = reason_steps + 1
                answer_seek_steps = answer_seek_steps + 1
                if reason_steps >= 24:
                    setup_ready = True
                if reason_steps >= 32:
                    late_ready = True
                if helpers.EndsWithLeftDelimiter(generated):
                    inside_span = True
                    phase = "span"
                    open_attempts = 0
                else:
                    phase = "answer_seek"
        elif phase == "done":
            break
        else:
            should_open = False
            if reason_steps >= 24:
                setup_ready = True
            if reason_steps >= 32:
                late_ready = True

            if setup_ready:
                if reason_steps >= 28:
                    should_open = True
                elif helpers.IsComplete(generated):
                    should_open = True
                elif late_ready and helpers.MinStepsToComplete(generated) <= 2:
                    should_open = True
                elif late_ready and helpers.ParserDistanceToComplete(generated) <= 2:
                    should_open = True
                elif late_ready and helpers.ValidContinuationCount(generated) <= 3:
                    should_open = True

            if should_open:
                generated, stepsLeft = helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft)
                open_attempts = 1
                answer_seek_steps = 1
                if helpers.EndsWithLeftDelimiter(generated):
                    inside_span = True
                    phase = "span"
                    open_attempts = 0
                else:
                    phase = "open"
            else:
                generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
                reason_steps = reason_steps + 1
                if reason_steps >= 24:
                    setup_ready = True
                if reason_steps >= 32:
                    late_ready = True
                if helpers.EndsWithLeftDelimiter(generated):
                    inside_span = True
                    phase = "span"
        if stepsLeft >= stepsLeftBeforeIteration:
            break
    remainingSteps = stepsLeft
    return generated, remainingSteps
