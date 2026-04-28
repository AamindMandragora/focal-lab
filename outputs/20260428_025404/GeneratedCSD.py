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
    # Generate a long free-form reasoning prefix first, then open one delimited constrained span, grow it until it is complete and sufficiently rich, and close immediately without any unconstrained tail.
    # CSD_RATIONALE_END
    phase = 0
    reasoning_steps = 0
    answer_steps = 0
    min_reasoning_steps = 40
    max_reasoning_steps = 72
    min_answer_steps = 3
    close_ready = 0

    # invariant helpers.lm == lm
    # invariant helpers.parser == parser
    # invariant lm.ValidTokensIdsLogits()
    # invariant 0 <= stepsLeft <= maxSteps
    # invariant |generated| + stepsLeft <= maxSteps
    # decreases stepsLeft
    while stepsLeft > 0 and phase < 4:
        if phase == 0 and reasoning_steps < min_reasoning_steps:
            generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
            reasoning_steps = reasoning_steps + 1
            if reasoning_steps >= min_reasoning_steps:
                phase = 1
        elif phase == 0 and reasoning_steps < max_reasoning_steps and helpers.HasBudget(stepsLeft, helpers.MinStepsToComplete(generated) + 2):
            generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
            reasoning_steps = reasoning_steps + 1
            if reasoning_steps >= max_reasoning_steps:
                phase = 1
        elif phase == 0:
            generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
            reasoning_steps = reasoning_steps + 1
            phase = 1
        elif phase == 1:
            generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)
            phase = 2
        elif phase == 2 and helpers.CanConstrain(generated):
            generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
            answer_steps = answer_steps + 1
            if answer_steps >= min_answer_steps:
                close_ready = 1
        elif phase == 2 and parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)) and helpers.CanExtendConstrained(generated) and answer_steps < min_answer_steps:
            generated, stepsLeft = helpers.AppendExtendConstrainedStep(prompt, generated, stepsLeft)
            answer_steps = answer_steps + 1
            if answer_steps >= min_answer_steps:
                close_ready = 1
        elif phase == 2 and parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)) and helpers.CanExtendConstrained(generated) and answer_steps < 8 and close_ready == 0:
            generated, stepsLeft = helpers.AppendExtendConstrainedStep(prompt, generated, stepsLeft)
            answer_steps = answer_steps + 1
            if answer_steps >= min_answer_steps:
                close_ready = 1
        elif phase == 2 and parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)) and close_ready == 1:
            generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
            phase = 3
        elif phase == 2 and parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)):
            generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
            phase = 3
        else:
            break
    remainingSteps = stepsLeft
    return generated, remainingSteps
