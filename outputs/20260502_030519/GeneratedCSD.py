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
    # Use a two-stage GSM policy: spend an initial block of budget on ordinary
    # unconstrained reasoning so the model can set up the word problem, then enter
    # an answer-seeking phase early enough that natural delimiter opening has
    # multiple chances before budget gets tight. Track span state explicitly with
    # `in_span` and count closed spans so only the first completed `<<...>>` span is
    # treated as the final answer span. Outside a span, continue ordinary reasoning
    # until either sufficient reasoning has occurred or parser/budget signals suggest
    # it is time to seek the final answer; once in answer-seeking mode, repeatedly
    # use the natural left-delimiter nudge until `<<` is actually opened. Inside the
    # span, never exit merely because constraining is currently unavailable; first
    # check completion, and whenever `helpers.IsComplete(generated) or
    # helpers.CanConstrain(generated)` holds, use
    # `AppendConstrainedOrRightDelimiterStep` so the model can either complete the
    # answer expression or naturally emit `>>`. After the first right delimiter, stop
    # decoding to avoid exposing an early local computation as the answer or adding
    # extra text after the final boxed span.
    # CSD_RATIONALE_END
    phase = "reason"
    in_span = False
    closed_spans = 0
    reasoning_steps = 0
    nudge_steps = 0
    answer_mode = False

    # invariant helpers.lm == lm
    # invariant helpers.parser == parser
    # invariant lm.ValidTokensIdsLogits()
    # invariant 0 <= stepsLeft <= maxSteps
    # invariant |generated| + stepsLeft <= maxSteps
    # decreases stepsLeft
    while stepsLeft > 0 and closed_spans == 0:
        stepsLeftBeforeIteration = stepsLeft
        if in_span:
            if helpers.EndsWithRightDelimiter(generated):
                in_span = False
                closed_spans = closed_spans + 1
                phase = "done"
                break
            elif helpers.IsComplete(generated) or helpers.CanConstrain(generated):
                phase = "span"
                generated, stepsLeft = helpers.AppendConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft)
            elif helpers.IsDead(generated):
                break
            else:
                generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
        else:
            if helpers.EndsWithLeftDelimiter(generated):
                in_span = True
                phase = "span"
                generated, stepsLeft = helpers.AppendConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft)
            else:
                distance = helpers.ParserDistanceToComplete(generated)
                min_steps = helpers.MinStepsToComplete(generated)
                continuation_count = helpers.ValidContinuationCount(generated)
                enough_reasoning = reasoning_steps >= 8
                parser_ready = distance <= 4 or min_steps <= 4
                budget_ready = stepsLeft <= 18
                if not answer_mode and (enough_reasoning or parser_ready or budget_ready):
                    answer_mode = True
                    phase = "seek_open"
                    generated, stepsLeft = helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft)
                    nudge_steps = nudge_steps + 1
                elif answer_mode and not helpers.EndsWithLeftDelimiter(generated):
                    if nudge_steps < 6 or continuation_count > 0:
                        phase = "seek_open"
                        generated, stepsLeft = helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft)
                        nudge_steps = nudge_steps + 1
                    else:
                        generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
                        reasoning_steps = reasoning_steps + 1
                elif helpers.IsDead(generated):
                    break
                else:
                    phase = "reason"
                    generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
                    reasoning_steps = reasoning_steps + 1
        if stepsLeft >= stepsLeftBeforeIteration:
            break
    remainingSteps = stepsLeft
    return generated, remainingSteps
