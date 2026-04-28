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
    # Use natural LM-chosen delimiter transitions: free-form reasoning allows or nudges a natural left delimiter, then a constrained span grows under grammar control and may close with a natural right delimiter only after completion. Explicit phase and close-policy state keep the span structured and budget-bounded.
    # CSD_RATIONALE_END
    phase = 0
    reasoningSteps = 0
    answerSteps = 0
    closeDelay = 0
    sawLeft = False
    sawRight = False
    # invariant helpers.lm == lm
    # invariant helpers.parser == parser
    # invariant lm.ValidTokensIdsLogits()
    # invariant 0 <= stepsLeft <= maxSteps
    # invariant |generated| + stepsLeft <= maxSteps
    # decreases stepsLeft
    while stepsLeft > 0 and not sawRight:
        if phase == 0:
            next_token = eosToken
            new_steps = stepsLeft
            if stepsLeft <= 4 or reasoningSteps >= 6:
                next_token, new_steps = helpers.UnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft)
            else:
                next_token, new_steps = helpers.UnconstrainedAllowLeftDelimiterStep(prompt, generated, stepsLeft)
            generated = generated + [next_token]
            stepsLeft = new_steps
            reasoningSteps = reasoningSteps + 1
            if next_token == LeftDelimiter or next_token == " <<":
                sawLeft = True
                phase = 1
        elif phase == 1 and helpers.CanConstrain(generated):
            suffix = helpers.LongestValidSuffix(generated)
            complete_now = parser.IsCompletePrefix(suffix)
            continuation_count = parser.ValidContinuationCount(suffix)
            distance = parser.ParserDistanceToComplete(suffix)
            if complete_now and closeDelay >= 1 and (continuation_count <= 1 or stepsLeft <= distance + 1 or answerSteps >= 4):
                next_token, new_steps = helpers.ConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft)
                generated = generated + [next_token]
                stepsLeft = new_steps
                if next_token == RightDelimiter or next_token == " >>":
                    sawRight = True
                    phase = 2
                else:
                    answerSteps = answerSteps + 1
                    closeDelay = closeDelay + 1
            else:
                next_token, new_steps = helpers.ConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft)
                generated = generated + [next_token]
                stepsLeft = new_steps
                if next_token == RightDelimiter or next_token == " >>":
                    sawRight = True
                    phase = 2
                else:
                    answerSteps = answerSteps + 1
                    suffix = helpers.LongestValidSuffix(generated)
                    if parser.IsCompletePrefix(suffix):
                        closeDelay = closeDelay + 1
        else:
            generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
    remainingSteps = stepsLeft
    return generated, remainingSteps
