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
    # Use an adaptive three-phase policy: spend a substantial free-form setup budget first, then open one final verified answer span, and inside that span use completion-aware constrained decoding with a minimum answer length before closing so compact multi-clause arithmetic expressions are less likely to truncate.
    # CSD_RATIONALE_END
    phase = 0
    reason_steps = 0
    answer_steps = 0
    min_reason_steps = 40
    target_reason_steps = 56
    min_answer_steps = 12
    max_answer_steps = 24
    pressure = 0
    close_ready = 0

    # invariant helpers.lm == lm
    # invariant helpers.parser == parser
    # invariant lm.ValidTokensIdsLogits()
    # invariant 0 <= stepsLeft <= maxSteps
    # invariant |generated| + stepsLeft <= maxSteps
    # decreases stepsLeft
    while stepsLeft > 0 and phase < 3:
        if phase == 0:
            need_for_answer = min_answer_steps + 2
            if reason_steps < min_reason_steps:
                generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
                reason_steps = reason_steps + 1
            elif not helpers.HasBudget(stepsLeft, need_for_answer):
                generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)
                phase = 1
            elif reason_steps < target_reason_steps and pressure < 2:
                generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
                reason_steps = reason_steps + 1
                if not helpers.HasBudget(stepsLeft, min_answer_steps + 4):
                    pressure = pressure + 1
            else:
                generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)
                phase = 1
        elif phase == 1:
            if helpers.CanConstrain(generated):
                generated, stepsLeft = helpers.AppendTopKConstrainedStep(prompt, generated, 1, stepsLeft)
                answer_steps = answer_steps + 1
                phase = 2
            else:
                break
        elif phase == 2 and parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)):
            if answer_steps >= min_answer_steps:
                close_ready = 1
            if close_ready > 0 and (answer_steps >= max_answer_steps or not helpers.CanExtendConstrained(generated) or not helpers.HasBudget(stepsLeft, 1)):
                generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
                phase = 3
            elif helpers.CanExtendConstrained(generated):
                generated, stepsLeft = helpers.AppendExtendConstrainedStep(prompt, generated, stepsLeft)
                answer_steps = answer_steps + 1
            else:
                generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
                phase = 3
        elif phase == 2 and helpers.CanConstrain(generated) and not parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)):
            if answer_steps < 4:
                generated, stepsLeft = helpers.AppendTopKConstrainedStep(prompt, generated, 1, stepsLeft)
            elif helpers.MinStepsToComplete(generated) + 1 >= stepsLeft:
                generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
            else:
                generated, stepsLeft = helpers.AppendTopKConstrainedStep(prompt, generated, 1, stepsLeft)
            answer_steps = answer_steps + 1
        else:
            break
    remainingSteps = stepsLeft
    return generated, remainingSteps
