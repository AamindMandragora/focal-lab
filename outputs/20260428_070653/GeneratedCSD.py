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
    # Use adaptive single-stream decoding: begin with free-form reasoning, deliberately open one verified span inside the loop, grow the constrained span until completion, optionally extend it when budget and continuation structure suggest a richer expression, then close it with an explicit right delimiter under a completion guard.
    # CSD_RATIONALE_END
    phase = 0
    opened_span = 0
    extended_once = 0
    post_close_steps = 0

    # invariant helpers.lm == lm
    # invariant helpers.parser == parser
    # invariant lm.ValidTokensIdsLogits()
    # invariant 0 <= stepsLeft <= maxSteps
    # invariant |generated| + stepsLeft <= maxSteps
    # decreases stepsLeft
    while stepsLeft > 0 and phase < 4:
        if phase == 0 and opened_span == 0 and len(generated) == 0:
            generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
        elif phase == 0 and opened_span == 0 and stepsLeft > helpers.MinStepsToComplete(generated) + 2:
            generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
            if len(generated) > 0:
                phase = 1
        elif phase == 0 and opened_span == 0:
            generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)
            opened_span = 1
            phase = 2
        elif phase == 1 and opened_span == 0:
            generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)
            opened_span = 1
            phase = 2
        elif phase == 2 and helpers.CanConstrain(generated):
            generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
        elif phase == 2 and parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)) and helpers.CanExtendConstrained(generated) and extended_once == 0 and stepsLeft > 1:
            generated, stepsLeft = helpers.AppendExtendConstrainedStep(prompt, generated, stepsLeft)
            extended_once = 1
        elif phase == 2 and parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)):
            suffix = helpers.LongestValidSuffix(generated)
            distance = parser.ParserDistanceToComplete(suffix)
            continuations = parser.ValidContinuationCount(suffix)
            if distance == 0 and (continuations <= 1 or stepsLeft <= 2 or extended_once > 0):
                generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
                phase = 3
            elif helpers.CanExtendConstrained(generated):
                generated, stepsLeft = helpers.AppendExtendConstrainedStep(prompt, generated, stepsLeft)
                extended_once = 1
            else:
                generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
                phase = 3
        elif phase == 3 and post_close_steps == 0:
            generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
            post_close_steps = 1
        elif phase == 3:
            break
        else:
            generated, stepsLeft = helpers.AppendBudgetAwareStep(prompt, generated, stepsLeft, 1)
    remainingSteps = stepsLeft
    return generated, remainingSteps
