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
    # Use free-form reasoning where the LM may naturally choose the left delimiter, then stay in a constrained span where the LM may continue the grammar or naturally choose the right delimiter only after completion. Budget pressure and span length control when to nudge opening and when to prefer closing.
    # CSD_RATIONALE_END
    phase = 0
    reasoningSteps = 0
    spanTokens = 0
    closePreference = 0
    openedNaturally = 0

    # invariant helpers.lm == lm
    # invariant helpers.parser == parser
    # invariant lm.ValidTokensIdsLogits()
    # invariant 0 <= stepsLeft <= maxSteps
    # invariant |generated| + stepsLeft <= maxSteps
    # decreases stepsLeft
    while stepsLeft > 0 and phase < 3:
        if phase == 0:
            next_token = eosToken
            new_steps = stepsLeft
            if stepsLeft <= 3 or reasoningSteps >= 6:
                next_token, new_steps = helpers.UnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft)
            else:
                next_token, new_steps = helpers.UnconstrainedAllowLeftDelimiterStep(prompt, generated, stepsLeft)
            generated = generated + [next_token]
            stepsLeft = new_steps
            reasoningSteps = reasoningSteps + 1
            if next_token == LeftDelimiter or next_token == " <<":
                phase = 1
                openedNaturally = 1
                spanTokens = 0
                closePreference = 0
        elif phase == 1 and helpers.CanConstrain(generated):
            suffix = helpers.LongestValidSuffix(generated)
            complete_now = parser.IsCompletePrefix(suffix)
            continuation_count = parser.ValidContinuationCount(suffix)
            distance = parser.ParserDistanceToComplete(suffix)
            next_token = eosToken
            new_steps = stepsLeft
            next_token, new_steps = helpers.ConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft)
            generated = generated + [next_token]
            stepsLeft = new_steps
            if next_token == RightDelimiter or next_token == " >>":
                phase = 2
            else:
                spanTokens = spanTokens + 1
                if complete_now:
                    closePreference = closePreference + 1
                elif distance <= 1:
                    closePreference = closePreference + 1
                else:
                    closePreference = 0
                if continuation_count <= 1 and spanTokens >= 2:
                    closePreference = closePreference + 1
        elif phase == 1 and parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)):
            next_token = eosToken
            new_steps = stepsLeft
            next_token, new_steps = helpers.ConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft)
            generated = generated + [next_token]
            stepsLeft = new_steps
            if next_token == RightDelimiter or next_token == " >>":
                phase = 2
            else:
                spanTokens = spanTokens + 1
                closePreference = closePreference + 1
        else:
            break
    remainingSteps = stepsLeft
    return generated, remainingSteps
