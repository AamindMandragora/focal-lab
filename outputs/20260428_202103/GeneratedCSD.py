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
    # Use free-form reasoning until the LM naturally opens a delimiter, then keep all span tokens grammar-controlled with constrained steps and ConstrainedOrRightDelimiterStep so complete expressions can close naturally instead of being extended indefinitely.
    # CSD_RATIONALE_END
    phase = 0
    closed_spans = 0
    reason_steps = 0
    final_ready = 0
    next_token = eosToken
    new_steps = stepsLeft
    # invariant helpers.lm == lm
    # invariant helpers.parser == parser
    # invariant lm.ValidTokensIdsLogits()
    # invariant 0 <= stepsLeft <= maxSteps
    # invariant |generated| + stepsLeft <= maxSteps
    # decreases stepsLeft
    while stepsLeft > 0 and phase < 3 and closed_spans < 3:
        if phase == 0:
            if closed_spans > 0 or reason_steps > 2:
                next_token, new_steps = helpers.UnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft)
            else:
                next_token, new_steps = helpers.UnconstrainedAllowLeftDelimiterStep(prompt, generated, stepsLeft)
            generated = generated + [next_token]
            stepsLeft = new_steps
            if next_token == LeftDelimiter or next_token == SpacedLeftDelimiter:
                phase = 1
            else:
                reason_steps = reason_steps + 1
                if closed_spans > 0 and reason_steps > 3:
                    final_ready = 1
        elif phase == 1 and helpers.CanConstrain(generated):
            generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
            reason_steps = reason_steps + 1
            if parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)):
                if final_ready > 0:
                    phase = 2
                elif closed_spans == 0 and reason_steps > 4:
                    phase = 2
        elif phase == 1 and parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)):
            next_token, new_steps = helpers.ConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft)
            generated = generated + [next_token]
            stepsLeft = new_steps
            if next_token == RightDelimiter or next_token == SpacedRightDelimiter:
                closed_spans = closed_spans + 1
                if final_ready > 0 or closed_spans >= 2:
                    phase = 3
                else:
                    phase = 0
                    reason_steps = 0
            else:
                reason_steps = reason_steps + 1
                if parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)):
                    if final_ready > 0:
                        phase = 2
                    elif parser.ValidContinuationCount(helpers.LongestValidSuffix(generated)) <= 1:
                        phase = 2
        elif phase == 2 and parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)):
            next_token, new_steps = helpers.ConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft)
            generated = generated + [next_token]
            stepsLeft = new_steps
            if next_token == RightDelimiter or next_token == SpacedRightDelimiter:
                closed_spans = closed_spans + 1
                phase = 3
            else:
                reason_steps = reason_steps + 1
        else:
            break
    remainingSteps = stepsLeft
    return generated, remainingSteps
