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
    # Use ordinary reasoning first, wait for stronger readiness signals or budget pressure, then let the LM naturally open one verified span and finish it with grammar-controlled tokens until a complete answer can naturally close.
    # CSD_RATIONALE_END
    phase = 0
    closed_spans = 0
    reason_steps = 0
    final_ready = 0
    saw_transition = 0
    next_token = eosToken
    new_steps = stepsLeft
    # invariant helpers.lm == lm
    # invariant helpers.parser == parser
    # invariant lm.ValidTokensIdsLogits()
    # invariant 0 <= stepsLeft <= maxSteps
    # invariant |generated| + stepsLeft <= maxSteps
    # decreases stepsLeft
    while stepsLeft > 0 and phase < 3 and closed_spans < 2:
        if phase == 0:
            if final_ready == 0 and helpers.HasBudget(stepsLeft, 18):
                generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
                reason_steps = reason_steps + 1
                if reason_steps >= 12:
                    saw_transition = 1
                if saw_transition > 0 and not helpers.HasBudget(stepsLeft, 12):
                    final_ready = 1
            else:
                next_token, new_steps = helpers.UnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft)
                generated = generated + [next_token]
                stepsLeft = new_steps
                if next_token == LeftDelimiter or next_token == SpacedLeftDelimiter:
                    phase = 1
                else:
                    reason_steps = reason_steps + 1
                    if saw_transition > 0 and not helpers.HasBudget(stepsLeft, 10):
                        final_ready = 1
        elif phase == 1 and (helpers.CanConstrain(generated) or parser.IsCompletePrefix(helpers.LongestValidSuffix(generated))):
            next_token, new_steps = helpers.ConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft)
            generated = generated + [next_token]
            stepsLeft = new_steps
            if next_token == RightDelimiter or next_token == SpacedRightDelimiter:
                closed_spans = closed_spans + 1
                phase = 3
            else:
                if parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)):
                    saw_transition = 1
        else:
            break
    remainingSteps = stepsLeft
    return generated, remainingSteps
