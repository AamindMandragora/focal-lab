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
    # Use ordinary unconstrained reasoning first, then naturally nudge the LM to open one verified delimiter span, generate grammar-valid answer tokens inside it, and let ConstrainedOrRightDelimiterStep choose either continuation or closure under an adaptive close policy.
    # CSD_RATIONALE_END
    phase = 0
    reason_steps = 0
    answer_ready = 0
    closed_spans = 0
    inside_steps = 0
    close_pressure = 0
    next_token = eosToken
    new_steps = stepsLeft
    # invariant helpers.lm == lm
    # invariant helpers.parser == parser
    # invariant lm.ValidTokensIdsLogits()
    # invariant 0 <= stepsLeft <= maxSteps
    # invariant |generated| + stepsLeft <= maxSteps
    # decreases stepsLeft
    while stepsLeft > 0 and phase < 3 and closed_spans < 2:
        if phase == 0 and answer_ready == 0 and helpers.HasBudget(stepsLeft, 12):
            generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
            reason_steps = reason_steps + 1
            if reason_steps >= 8:
                answer_ready = 1
        elif phase == 0:
            next_token, new_steps = helpers.UnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft)
            generated = generated + [next_token]
            stepsLeft = new_steps
            if next_token == LeftDelimiter or next_token == SpacedLeftDelimiter:
                phase = 1
                inside_steps = 0
                close_pressure = 0
            else:
                reason_steps = reason_steps + 1
                if reason_steps >= 12:
                    answer_ready = 1
        elif phase == 1 and helpers.CanConstrain(generated):
            generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
            inside_steps = inside_steps + 1
            if inside_steps >= 3:
                close_pressure = 1
        elif phase == 1 and parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)):
            if close_pressure > 0 or not helpers.HasBudget(stepsLeft, 3) or parser.ValidContinuationCount(helpers.LongestValidSuffix(generated)) <= 1:
                next_token, new_steps = helpers.ConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft)
                generated = generated + [next_token]
                stepsLeft = new_steps
                if next_token == RightDelimiter or next_token == SpacedRightDelimiter:
                    closed_spans = closed_spans + 1
                    phase = 2
                else:
                    inside_steps = inside_steps + 1
                    close_pressure = 1
            else:
                next_token, new_steps = helpers.ConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft)
                generated = generated + [next_token]
                stepsLeft = new_steps
                if next_token == RightDelimiter or next_token == SpacedRightDelimiter:
                    closed_spans = closed_spans + 1
                    phase = 2
                else:
                    inside_steps = inside_steps + 1
                    if inside_steps >= 4:
                        close_pressure = 1
        elif phase == 2:
            break
        else:
            break
    remainingSteps = stepsLeft
    return generated, remainingSteps
