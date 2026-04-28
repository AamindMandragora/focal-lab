from __future__ import annotations

from VerifiedAgentSynthesis import (
    CSDHelpers,
    LM,
    LeftDelimiter,
    RightDelimiter,
    SpacedLeftDelimiter,
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
    # Use free-form reasoning first, but let the LM naturally choose when to open a delimited answer span.
    # Once inside a span, keep tokens grammar-controlled and let the LM naturally choose the right delimiter only after completion, with an adaptive close policy based on answer richness and budget.
    # CSD_RATIONALE_END
    phase = 0
    reasoningSteps = 0
    answerSteps = 0
    closeBias = 0
    sawLeft = 0
    sawRight = 0

    # invariant helpers.lm == lm
    # invariant helpers.parser == parser
    # invariant lm.ValidTokensIdsLogits()
    # invariant 0 <= stepsLeft <= maxSteps
    # invariant |generated| + stepsLeft <= maxSteps
    # decreases stepsLeft
    while stepsLeft > 0 and sawRight == 0:
        if phase == 0:
            next_token, new_steps = helpers.UnconstrainedAllowLeftDelimiterStep(prompt, generated, stepsLeft)
            generated = generated + [next_token]
            stepsLeft = new_steps
            reasoningSteps = reasoningSteps + 1
            if next_token == "<<":
                sawLeft = 1
                phase = 2
            elif next_token == eosToken:
                break
            elif reasoningSteps >= 6 and stepsLeft > 0:
                generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)
                sawLeft = 1
                phase = 2
            else:
                phase = 0
        elif phase == 2 and helpers.CanConstrain(generated):
            generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
            answerSteps = answerSteps + 1
            if answerSteps >= 2:
                closeBias = 1
            else:
                closeBias = 0
            phase = 2
        elif phase == 2 and parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)):
            suffix = helpers.LongestValidSuffix(generated)
            continuation_count = parser.ValidContinuationCount(suffix)
            distance = parser.ParserDistanceToComplete(suffix)
            next_token = eosToken
            new_steps = stepsLeft
            if (closeBias > 0 and continuation_count <= 1) or stepsLeft <= 2 or answerSteps >= 5 or distance == 0:
                next_token, new_steps = helpers.ConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft)
                generated = generated + [next_token]
                stepsLeft = new_steps
                if next_token == ">>":
                    sawRight = 1
                    phase = 3
                else:
                    answerSteps = answerSteps + 1
                    phase = 2
            elif phase == 2 and helpers.CanExtendConstrained(generated):
                generated, stepsLeft = helpers.AppendExtendConstrainedStep(prompt, generated, stepsLeft)
                answerSteps = answerSteps + 1
                closeBias = closeBias + 1
                phase = 2
            else:
                next_token, new_steps = helpers.ConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft)
                generated = generated + [next_token]
                stepsLeft = new_steps
                if next_token == ">>":
                    sawRight = 1
                    phase = 3
                else:
                    answerSteps = answerSteps + 1
                    phase = 2
        elif phase == 2:
            break
        else:
            break
    remainingSteps = stepsLeft
    return generated, remainingSteps
