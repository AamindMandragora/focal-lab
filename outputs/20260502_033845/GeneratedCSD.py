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
    # Use natural-delimiter decoding with explicit span-state tracking, explicit
    # closed-span counting, and a scratch-vs-final decision at right-delimiter
    # events. Ordinary reasoning remains unconstrained. When the decoder appears
    # ready for an early non-final verified span, mark scratch intent so the next
    # opened span is treated as a scratch span. Once globally answer-ready, stop
    # creating scratch intent and repeatedly nudge toward opening the final span.
    # Inside any open span, only continue when the positive guard
    # helpers.IsComplete(generated) or helpers.CanConstrain(generated) holds, then
    # advance with the constrained-or-close helper until a right delimiter appears.
    # Crucially, closing a span does not immediately terminate unless that closed
    # span is the final span; scratch spans instead transition back to ordinary
    # reasoning so additional spans may follow.
    # CSD_RATIONALE_END
    phase = "reason"
    inside_span = False
    closed_spans = 0
    nudge_mode = False
    final_ready = False
    scratch_mode = False
    scratch_ready = False
    opening_scratch_span = False
    current_span_is_final = False

    # invariant helpers.lm == lm
    # invariant helpers.parser == parser
    # invariant lm.ValidTokensIdsLogits()
    # invariant 0 <= stepsLeft <= maxSteps
    # invariant |generated| + stepsLeft <= maxSteps
    # decreases stepsLeft
    while stepsLeft > 0:
        stepsLeftBeforeIteration = stepsLeft

        if inside_span:
            if helpers.EndsWithRightDelimiter(generated):
                inside_span = False
                phase = "post_span"
                closed_spans = closed_spans + 1
                if opening_scratch_span:
                    scratch_mode = False
                    scratch_ready = False
                    opening_scratch_span = False
                if current_span_is_final and final_ready:
                    break
                current_span_is_final = False
                generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
            elif helpers.IsComplete(generated) or helpers.CanConstrain(generated):
                generated, stepsLeft = helpers.AppendConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft)
                if helpers.EndsWithRightDelimiter(generated):
                    inside_span = False
                    phase = "post_span"
                    closed_spans = closed_spans + 1
                    if opening_scratch_span:
                        scratch_mode = False
                        scratch_ready = False
                        opening_scratch_span = False
                    if current_span_is_final and final_ready:
                        break
                    current_span_is_final = False
            else:
                break
        elif helpers.EndsWithLeftDelimiter(generated):
            inside_span = True
            phase = "span"
            current_span_is_final = final_ready and not scratch_ready
            if opening_scratch_span:
                scratch_mode = True
                scratch_ready = False
                current_span_is_final = False
            generated, stepsLeft = helpers.AppendConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft)
            if helpers.EndsWithRightDelimiter(generated):
                inside_span = False
                phase = "post_span"
                closed_spans = closed_spans + 1
                if opening_scratch_span:
                    scratch_mode = False
                    scratch_ready = False
                    opening_scratch_span = False
                if current_span_is_final and final_ready:
                    break
                current_span_is_final = False
        else:
            if closed_spans >= 1 and helpers.MinStepsToComplete(generated) <= 4:
                final_ready = True
            elif closed_spans >= 2 and helpers.ParserDistanceToComplete(generated) <= 3:
                final_ready = True
            elif closed_spans >= 1 and helpers.ValidContinuationCount(generated) <= 2:
                final_ready = True

            if final_ready:
                scratch_ready = False
                scratch_mode = False
                opening_scratch_span = False
                nudge_mode = True
                generated, stepsLeft = helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft)
                if helpers.EndsWithLeftDelimiter(generated):
                    inside_span = True
                    phase = "span"
                    nudge_mode = False
                    opening_scratch_span = False
                    current_span_is_final = True
            elif nudge_mode:
                generated, stepsLeft = helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft)
                if helpers.EndsWithLeftDelimiter(generated):
                    inside_span = True
                    phase = "span"
                    nudge_mode = False
                    opening_scratch_span = scratch_ready
                    current_span_is_final = False
                    if opening_scratch_span:
                        scratch_mode = True
                        scratch_ready = False
            else:
                generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
                if closed_spans == 0 and helpers.ParserDistanceToComplete(generated) <= 6:
                    nudge_mode = True
                    scratch_ready = True
                    opening_scratch_span = False
                elif closed_spans >= 1 and helpers.ValidContinuationCount(generated) <= 4 and not final_ready:
                    nudge_mode = True
                    scratch_ready = True
                    opening_scratch_span = False
                elif closed_spans >= 1 and helpers.MinStepsToComplete(generated) <= 5 and not final_ready:
                    nudge_mode = True
                    scratch_ready = True
                    opening_scratch_span = False

        if stepsLeft >= stepsLeftBeforeIteration:
            break
    remainingSteps = stepsLeft
    return generated, remainingSteps
