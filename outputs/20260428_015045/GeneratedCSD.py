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
    # Use a longer free-form reasoning phase to let the model solve the word problem naturally, then
    # emit one compact verified answer span near the end. Inside the span, force hard grammar control,
    # prefer immediate completion when budget is tight, and allow only a small amount of extension so
    # the final span is usually a short arithmetic expression rather than a long drifting derivation.
    # CSD_RATIONALE_END
    phase = 0
    reason_steps = 0
    max_reason_steps = 24
    min_reason_steps = 8
    answer_steps = 0
    min_answer_steps = 3
    max_answer_steps = 12
    close_slack = 2

    # invariant helpers.lm == lm
    # invariant helpers.parser == parser
    # invariant lm.ValidTokensIdsLogits()
    # invariant 0 <= stepsLeft <= maxSteps
    # invariant |generated| + stepsLeft <= maxSteps
    # decreases stepsLeft
    while stepsLeft > 0 and phase < 3:
        if phase == 0:
            need_for_answer = 1
            if helpers.HasBudget(stepsLeft, close_slack + 2):
                need_for_answer = 0

            if reason_steps < min_reason_steps and need_for_answer == 0:
                generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
                reason_steps = reason_steps + 1
            elif reason_steps < max_reason_steps and need_for_answer == 0 and stepsLeft > helpers.MinStepsToComplete(generated) + close_slack + 1:
                generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
                reason_steps = reason_steps + 1
            else:
                generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)
                phase = 1

        elif phase == 1 and helpers.CanConstrain(generated):
            if stepsLeft <= close_slack + 2:
                generated, stepsLeft = helpers.AppendTopKConstrainedStep(prompt, generated, 1, stepsLeft)
            else:
                generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
            answer_steps = answer_steps + 1

        elif phase == 1 and parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)):
            suffix = helpers.LongestValidSuffix(generated)
            continuation_count = parser.ValidContinuationCount(suffix)
            should_close = 0

            if answer_steps >= min_answer_steps:
                should_close = 1
            if continuation_count == 0:
                should_close = 1
            if answer_steps >= max_answer_steps:
                should_close = 1
            if stepsLeft <= 1:
                should_close = 1

            if should_close == 1:
                generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
                phase = 2
            elif helpers.CanExtendConstrained(generated):
                generated, stepsLeft = helpers.AppendExtendConstrainedStep(prompt, generated, stepsLeft)
                answer_steps = answer_steps + 1
            else:
                generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
                phase = 2

        elif phase == 2:
            if stepsLeft > 0:
                generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
            else:
                break

        else:
            break
    remainingSteps = stepsLeft
    return generated, remainingSteps
