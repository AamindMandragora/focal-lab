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
    # Strategy: use an explicit phase machine with durable setup before any parser-
    # readiness signal can influence answer opening. First spend a long reasoning
    # phase generating ordinary GSM-style scratch work only. After that durable
    # setup, allow an "answer_ready" transition only when either the setup is very
    # long or parser-shape signals appear after the long setup / final-cue phase.
    # Then repeatedly nudge toward a natural left delimiter until one is emitted.
    # Once inside the answer span, use the positive guard
    # `helpers.IsComplete(generated) or helpers.CanConstrain(generated)` and consume
    # `AppendConstrainedOrRightDelimiterStep` so the model can either continue the
    # answer or close with `>>`. Closed spans are tracked explicitly, and decoding
    # stops after one closed answer span.
    # CSD_RATIONALE_END
    phase = "reason"
    in_span = False
    closed_spans = 0
    reason_steps = 0
    open_attempts = 0
    answer_ready = False

    durable_setup_steps = 44
    very_long_setup_steps = 60
    max_open_attempts = 16

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
        elif in_span:
            phase = "span"
            if helpers.EndsWithRightDelimiter(generated):
                in_span = False
                closed_spans = closed_spans + 1
                phase = "done"
                break
            elif helpers.IsComplete(generated) or helpers.CanConstrain(generated):
                generated, stepsLeft = helpers.AppendConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft)
                if helpers.EndsWithRightDelimiter(generated):
                    in_span = False
                    closed_spans = closed_spans + 1
                    phase = "done"
                else:
                    in_span = True
                    phase = "span"
            else:
                generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
                if helpers.EndsWithRightDelimiter(generated):
                    in_span = False
                    closed_spans = closed_spans + 1
                    phase = "done"
                else:
                    in_span = True
                    phase = "span"
        elif phase == "reason":
            if reason_steps >= very_long_setup_steps:
                answer_ready = True
            elif reason_steps >= durable_setup_steps:
                if helpers.IsComplete(generated):
                    answer_ready = True
                elif helpers.CanConstrain(generated) and helpers.ValidContinuationCount(generated) <= 2:
                    answer_ready = True
                elif helpers.ParserDistanceToComplete(generated) <= 2 and helpers.MinStepsToComplete(generated) <= 2:
                    answer_ready = True

            if answer_ready:
                phase = "open"
                generated, stepsLeft = helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft)
                open_attempts = open_attempts + 1
                if helpers.EndsWithLeftDelimiter(generated):
                    in_span = True
                    phase = "span"
                else:
                    in_span = False
                    phase = "open"
            else:
                generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
                reason_steps = reason_steps + 1
                if helpers.EndsWithLeftDelimiter(generated):
                    in_span = True
                    phase = "span"
                else:
                    phase = "reason"
        elif phase == "open":
            if helpers.EndsWithLeftDelimiter(generated):
                in_span = True
                phase = "span"
                generated, stepsLeft = helpers.AppendConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft)
                if helpers.EndsWithRightDelimiter(generated):
                    in_span = False
                    closed_spans = closed_spans + 1
                    phase = "done"
                else:
                    in_span = True
                    phase = "span"
            elif open_attempts >= max_open_attempts:
                generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
                if helpers.EndsWithLeftDelimiter(generated):
                    in_span = True
                    phase = "span"
                else:
                    phase = "open"
            else:
                generated, stepsLeft = helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft)
                open_attempts = open_attempts + 1
                if helpers.EndsWithLeftDelimiter(generated):
                    in_span = True
                    phase = "span"
                else:
                    in_span = False
                    phase = "open"
        elif phase == "span":
            if helpers.EndsWithRightDelimiter(generated):
                in_span = False
                closed_spans = closed_spans + 1
                phase = "done"
                break
            elif helpers.IsComplete(generated) or helpers.CanConstrain(generated):
                generated, stepsLeft = helpers.AppendConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft)
                if helpers.EndsWithRightDelimiter(generated):
                    in_span = False
                    closed_spans = closed_spans + 1
                    phase = "done"
                else:
                    in_span = True
                    phase = "span"
            else:
                generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
        else:
            break
    remainingSteps = stepsLeft
    return generated, remainingSteps
