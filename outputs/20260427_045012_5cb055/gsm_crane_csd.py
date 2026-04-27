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
    # This strategy aims to balance free-form reasoning and grammar-constrained output within the << >> delimiters.
    # It starts with a brief period of free-form reasoning followed by constrained grammar-based generation.
    # If the grammar constraints become too restrictive, it falls back to unconstrained generation.
    # CSD_RATIONALE_END
    phase = 0
    reasoning_steps = 0
    constrained_steps = 0
    reasoning_budget = 4

    # Ensure the initial steps are used for free-form reasoning
    if stepsLeft < 10:
        reasoning_budget = stepsLeft // 2

    # invariant lm.ValidTokensIdsLogits()
    # invariant 0 <= stepsLeft <= maxSteps
    # invariant 0 <= reasoning_steps
    # invariant 0 <= constrained_steps
    # invariant 0 <= phase <= 3
    # invariant |generated| + stepsLeft <= maxSteps
    # decreases stepsLeft
    while stepsLeft > 0 and phase < 3:
        if phase == 0 and reasoning_steps < reasoning_budget and stepsLeft > 2:
            generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
            reasoning_steps += 1
            if reasoning_steps >= reasoning_budget or stepsLeft <= 2:
                phase = 1
        elif phase == 0:
            generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)
            phase = 2
        elif phase == 1:
            generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)
            phase = 2
        elif phase == 2 and helpers.CanConstrain(generated):
            generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
            constrained_steps += 1
        elif phase == 2 and not helpers.CanConstrain(generated):
            # Fall back to unconstrained generation if grammar constraints are too restrictive
            # invariant lm.ValidTokensIdsLogits()
            # invariant 0 <= stepsLeft <= maxSteps
            # invariant |generated| + stepsLeft <= maxSteps
            # decreases stepsLeft
            while stepsLeft > 0 and phase == 2:
                generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
                constrained_steps += 1
            phase = 3
        elif phase == 3:
            generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
            break
    remainingSteps = stepsLeft
    return generated, remainingSteps
