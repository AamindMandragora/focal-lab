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
    # Use a single final verified span instead of an early scratch span. Spend most budget on
    # unconstrained chain-of-thought, then open one constrained answer span late and keep extending
    # it while complete if budget allows, encouraging a richer arithmetic expression rather than an
    # early short numeral.
    # CSD_RATIONALE_END
    phase = 0
    reason_steps = 0
    answer_steps = 0
    min_reason = 8
    max_reason = 40
    close_ready = 0

    # invariant helpers.lm == lm
    # invariant helpers.parser == parser
    # invariant lm.ValidTokensIdsLogits()
    # invariant 0 <= stepsLeft <= maxSteps
    # invariant |generated| + stepsLeft <= maxSteps
    # decreases stepsLeft
    while stepsLeft > 0 and phase < 3:
        if phase == 0:
            if reason_steps < min_reason and stepsLeft > helpers.MinStepsToComplete(generated) + 3:
                generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
                reason_steps = reason_steps + 1
            elif reason_steps < max_reason and stepsLeft > helpers.MinStepsToComplete(generated) + 6:
                generated, stepsLeft = helpers.AppendBudgetAwareStep(prompt, generated, stepsLeft, 2)
                reason_steps = reason_steps + 1
            else:
                generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)
                phase = 1
        elif phase == 1 and helpers.CanConstrain(generated):
            if answer_steps == 0:
                generated, stepsLeft = helpers.AppendTopKConstrainedStep(prompt, generated, 1, stepsLeft)
            else:
                generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
            answer_steps = answer_steps + 1
        elif phase == 1 and parser.IsCompletePrefix(helpers.LongestValidSuffix(generated)):
            if helpers.CanExtendConstrained(generated) and answer_steps < 12 and stepsLeft > 1 and close_ready < 2:
                generated, stepsLeft = helpers.AppendExtendConstrainedStep(prompt, generated, stepsLeft)
                answer_steps = answer_steps + 1
                close_ready = close_ready + 1
            else:
                generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
                phase = 2
        else:
            break
    remainingSteps = stepsLeft
    return generated, remainingSteps
