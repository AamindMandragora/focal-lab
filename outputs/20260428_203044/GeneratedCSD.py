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
    # Adaptive GSM-symbolic strategy: begin with delimiter-masked free-form reasoning, then let the LM
    # naturally open verified spans after a reasoning cue or budget pressure. Inside each span, use
    # grammar-controlled decoding with natural right-delimiter choice, allow one scratch span before a
    # final span, and stop only after a closed final verified expression.
    # CSD_RATIONALE_END
    phase = 0
    closed_spans = 0
    reason_signal = 0
    final_ready = 0
    scratch_goal = 1
    close_ready = 0
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
            if final_ready == 0 and closed_spans == 0 and helpers.HasBudget(stepsLeft, 18):
                generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
                reason_signal = reason_signal + 1
                if reason_signal >= 6:
                    final_ready = 1
            elif final_ready == 0 and closed_spans > 0 and helpers.HasBudget(stepsLeft, 12):
                generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
                reason_signal = reason_signal + 1
                if reason_signal >= 9:
                    final_ready = 1
            else:
                next_token = eosToken
                new_steps = stepsLeft
                if helpers.HasBudget(stepsLeft, 10):
                    next_token, new_steps = helpers.UnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft)
                else:
                    next_token, new_steps = helpers.UnconstrainedAllowLeftDelimiterStep(prompt, generated, stepsLeft)
                generated = generated + [next_token]
                stepsLeft = new_steps
                if next_token == LeftDelimiter or next_token == " <<":
                    phase = 1
                    close_ready = 0
                else:
                    reason_signal = reason_signal + 1
                    if closed_spans > 0:
                        final_ready = 1
        elif phase == 1 and parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)):
            if close_ready > 0 or closed_spans >= scratch_goal or not helpers.CanExtendConstrained(generated) or not helpers.HasBudget(stepsLeft, 2):
                next_token, new_steps = helpers.ConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft)
                generated = generated + [next_token]
                stepsLeft = new_steps
                if next_token == RightDelimiter or next_token == " >>":
                    closed_spans = closed_spans + 1
                    reason_signal = 0
                    close_ready = 0
                    if final_ready > 0 and closed_spans > scratch_goal:
                        phase = 2
                    else:
                        phase = 0
                        if closed_spans >= scratch_goal:
                            final_ready = 1
                else:
                    close_ready = close_ready + 1
            elif helpers.CanExtendConstrained(generated):
                next_token, new_steps = helpers.ConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft)
                generated = generated + [next_token]
                stepsLeft = new_steps
                if next_token == RightDelimiter or next_token == " >>":
                    closed_spans = closed_spans + 1
                    reason_signal = 0
                    close_ready = 0
                    if final_ready > 0 and closed_spans > scratch_goal:
                        phase = 2
                    else:
                        phase = 0
                        if closed_spans >= scratch_goal:
                            final_ready = 1
                else:
                    close_ready = close_ready + 1
            else:
                break
        elif phase == 1 and helpers.CanConstrain(generated):
            next_token, new_steps = helpers.ConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft)
            generated = generated + [next_token]
            stepsLeft = new_steps
            if next_token == RightDelimiter or next_token == " >>":
                closed_spans = closed_spans + 1
                reason_signal = 0
                close_ready = 0
                if final_ready > 0 and closed_spans > scratch_goal:
                    phase = 2
                else:
                    phase = 0
                    if closed_spans >= scratch_goal:
                        final_ready = 1
            else:
                if parser.ParserDistanceToComplete(helpers.LongestValidSuffix(generated)) <= 1:
                    close_ready = close_ready + 1
        else:
            break
    remainingSteps = stepsLeft
    return generated, remainingSteps
