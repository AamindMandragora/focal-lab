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
    # Use adaptive free-form reasoning before any span, then naturally nudge/open a verified span after a cue or budget pressure. Inside spans, use grammar-controlled tokens with natural right-delimiter choice, allow scratch spans, return to reasoning after non-final closure, and stop after a likely final verified span.
    # CSD_RATIONALE_END
    phase = 0
    closed_spans = 0
    reason_signal = 0
    cue_signal = 0
    final_ready = 0
    close_ready = 0
    post_span_reason = 0
    next_token = eosToken
    new_steps = stepsLeft
    # invariant helpers.lm == lm
    # invariant helpers.parser == parser
    # invariant lm.ValidTokensIdsLogits()
    # invariant 0 <= stepsLeft <= maxSteps
    # invariant |generated| + stepsLeft <= maxSteps
    # decreases stepsLeft
    while stepsLeft > 0 and phase < 3 and closed_spans < 4:
        if phase == 0:
            if final_ready == 0 and helpers.HasBudget(stepsLeft, 18) and cue_signal == 0:
                generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
                reason_signal = reason_signal + 1
                if reason_signal >= 4:
                    cue_signal = 1
                if closed_spans > 0 and reason_signal > post_span_reason:
                    final_ready = 1
            else:
                next_token, new_steps = helpers.UnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft)
                generated = generated + [next_token]
                stepsLeft = new_steps
                if next_token == LeftDelimiter or next_token == " <<":
                    phase = 1
                    close_ready = 0
                else:
                    reason_signal = reason_signal + 1
                    if cue_signal == 0:
                        cue_signal = 1
                    if closed_spans > 0:
                        final_ready = 1
        elif phase == 1 and parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)):
            if close_ready > 0 or final_ready > 0 or not helpers.CanExtendConstrained(generated) or not helpers.HasBudget(stepsLeft, helpers.MinStepsToComplete(generated) + 2):
                next_token, new_steps = helpers.ConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft)
                generated = generated + [next_token]
                stepsLeft = new_steps
                if next_token == RightDelimiter or next_token == " >>":
                    closed_spans = closed_spans + 1
                    if final_ready > 0 or closed_spans >= 2:
                        phase = 3
                    else:
                        phase = 0
                        reason_signal = 0
                        cue_signal = 1
                        post_span_reason = 2
                else:
                    close_ready = close_ready + 1
            elif helpers.CanExtendConstrained(generated):
                next_token, new_steps = helpers.ConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft)
                generated = generated + [next_token]
                stepsLeft = new_steps
                if next_token == RightDelimiter or next_token == " >>":
                    closed_spans = closed_spans + 1
                    if final_ready > 0 or closed_spans >= 2:
                        phase = 3
                    else:
                        phase = 0
                        reason_signal = 0
                        cue_signal = 1
                        post_span_reason = 2
                else:
                    close_ready = close_ready + 1
            else:
                break
        elif phase == 1 and helpers.CanConstrain(generated):
            next_token, new_steps = helpers.ConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft)
            generated = generated + [next_token]
            stepsLeft = new_steps
            close_ready = 0
        else:
            break
    remainingSteps = stepsLeft
    return generated, remainingSteps
