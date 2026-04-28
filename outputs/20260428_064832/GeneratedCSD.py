from __future__ import annotations

from VerifiedAgentSynthesis import (
    CSDHelpers,
    LM,
    LeftDelimiter,
    RightDelimiter,
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
    # Generate a short amount of free-form reasoning first, then deliberately open one verified
    # delimiter span. Inside the span, keep all tokens grammar-controlled, use hard constrained
    # decoding while incomplete, optionally extend a completed answer a little, and close only when
    # completion is true and the close policy says the span is rich enough or budget is tight.
    # CSD_RATIONALE_END
    phase = 0
    reasonCount = 0
    spanTokens = 0
    milestoneSeen = 0
    recentCue = 0
    lastToken = ""
    next_token = eosToken
    new_steps = stepsLeft

    # invariant helpers.lm == lm
    # invariant helpers.parser == parser
    # invariant lm.ValidTokensIdsLogits()
    # invariant 0 <= stepsLeft <= maxSteps
    # invariant |generated| + stepsLeft <= maxSteps
    # decreases stepsLeft
    while stepsLeft > 0 and phase < 3:
        if phase == 0:
            shouldOpen = 0
            if milestoneSeen > 0 and reasonCount >= 6:
                shouldOpen = 1
            elif reasonCount >= 18:
                shouldOpen = 1
            elif stepsLeft <= 8:
                shouldOpen = 1

            if shouldOpen > 0:
                generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)
                phase = 1
            else:
                next_token, new_steps = helpers.UnconstrainedStep(prompt, generated, stepsLeft)
                generated = generated + [next_token]
                stepsLeft = new_steps
                reasonCount = reasonCount + 1
                lastToken = next_token

                if next_token == "." or next_token == ":" or next_token == ";" or next_token == " not " or next_token == "?" or next_token == "NL":
                    milestoneSeen = 1
                    recentCue = 0
                elif next_token == "therefore" or next_token == "Thus" or next_token == "thus" or next_token == "so" or next_token == "total" or next_token == "Total" or next_token == "answer" or next_token == "Answer" or next_token == "=":
                    recentCue = 1
                elif recentCue > 0:
                    if next_token == "is" or next_token == ":" or next_token == "=":
                        milestoneSeen = 1
                    recentCue = 0
        elif phase == 1 and helpers.CanConstrain(generated):
            generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
            spanTokens = spanTokens + 1
            phase = 2
        elif phase == 1:
            break
        elif phase == 2 and helpers.CanConstrain(generated):
            generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
            spanTokens = spanTokens + 1
        elif phase == 2 and parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)):
            suffix = helpers.LongestValidSuffix(generated)
            continuationCount = parser.ValidContinuationCount(suffix)
            distance = parser.ParserDistanceToComplete(suffix)
            closeNow = 0

            if not helpers.CanExtendConstrained(generated):
                closeNow = 1
            elif stepsLeft <= distance + 1:
                closeNow = 1
            elif spanTokens >= 9:
                closeNow = 1
            elif continuationCount <= 1 and spanTokens >= 3:
                closeNow = 1

            if closeNow > 0:
                generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
                phase = 3
            elif helpers.CanExtendConstrained(generated):
                generated, stepsLeft = helpers.AppendExtendConstrainedStep(prompt, generated, stepsLeft)
                spanTokens = spanTokens + 1
            else:
                generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
                phase = 3
        else:
            break
    remainingSteps = stepsLeft
    return generated, remainingSteps
