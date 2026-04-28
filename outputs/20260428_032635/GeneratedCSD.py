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
    # Use a long free-form reasoning phase with at least 40 unconstrained steps, then open one
    # delimited constrained span, grow it with hard constrained steps and optional extension,
    # and close only after completion plus a minimum answer-length policy.
    # CSD_RATIONALE_END
    phase = 0
    reasoning_steps = 0
    answer_steps = 0
    min_reasoning_steps = 40
    max_reasoning_steps = 64
    min_answer_steps = 3
    close_ready = 0

    # invariant helpers.lm == lm
    # invariant helpers.parser == parser
    # invariant lm.ValidTokensIdsLogits()
    # invariant 0 <= stepsLeft <= maxSteps
    # invariant |generated| + stepsLeft <= maxSteps
    # decreases stepsLeft
    while stepsLeft > 0 and phase < 3:
        if phase == 0 and reasoning_steps < min_reasoning_steps:
            generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
            reasoning_steps = reasoning_steps + 1
        elif phase == 0 and reasoning_steps < max_reasoning_steps and helpers.HasBudget(stepsLeft, helpers.MinStepsToComplete(generated) + min_answer_steps + 2):
            generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
            reasoning_steps = reasoning_steps + 1
        elif phase == 0:
            generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)
            phase = 1
        elif phase == 1 and helpers.CanConstrain(generated):
            generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
            answer_steps = answer_steps + 1
            phase = 2
        elif phase == 1 and parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)):
            if answer_steps >= min_answer_steps:
                generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
                phase = 3
            elif helpers.CanExtendConstrained(generated):
                generated, stepsLeft = helpers.AppendExtendConstrainedStep(prompt, generated, stepsLeft)
                answer_steps = answer_steps + 1
                phase = 2
            else:
                generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
                phase = 3
        elif phase == 2 and helpers.CanConstrain(generated):
            generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
            answer_steps = answer_steps + 1
        elif phase == 2 and parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)) and answer_steps < min_answer_steps and helpers.CanExtendConstrained(generated):
            generated, stepsLeft = helpers.AppendExtendConstrainedStep(prompt, generated, stepsLeft)
            answer_steps = answer_steps + 1
        elif phase == 2 and parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)):
            close_ready = 1
            if close_ready == 1:
                generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
                phase = 3
            else:
                break
        else:
            break
    remainingSteps = stepsLeft
    return generated, remainingSteps
