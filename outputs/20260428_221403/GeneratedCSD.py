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
    # Use one loop with explicit span-state tracking so closed verified spans are
    # counted across the whole generation. This prevents the first short verified
    # mini-expression from being mistaken for the final answer.
    # Start in free-form reasoning with delimiter masking. After enough ordinary
    # reasoning, or under budget pressure, switch to nudging a natural left
    # delimiter to open a scratch span. After a scratch span closes, continue in
    # free-form mode so later reasoning can compose that scratch result into a final
    # verified span. The strategy allows multiple spans, but only terminates after a
    # later closed span once at least one prior span has already closed.
    # Inside a span, use constrained-or-right-delimiter steps. If the suffix is not
    # yet complete, continue constraining; if it is complete, allow either a valid
    # continuation or a natural right delimiter. Every non-break branch consumes
    # exactly one helper step and updates stepsLeft from the helper return.
    # CSD_RATIONALE_END
    phase = 0
    free_steps = 0
    closed_spans = 0
    post_span_free_steps = 0

    # invariant helpers.lm == lm
    # invariant helpers.parser == parser
    # invariant lm.ValidTokensIdsLogits()
    # invariant 0 <= stepsLeft <= maxSteps
    # invariant |generated| + stepsLeft <= maxSteps
    # decreases stepsLeft
    while stepsLeft > 0:
        if phase == 2:
            if helpers.EndsWithRightDelimiter(generated):
                closed_spans = closed_spans + 1
                if closed_spans >= 2:
                    break
                phase = 0
                post_span_free_steps = 0
                generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
                free_steps = free_steps + 1
                post_span_free_steps = post_span_free_steps + 1
            elif not helpers.CanConstrain(generated):
                break
            elif helpers.IsComplete(generated):
                generated, stepsLeft = helpers.AppendConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft)
            else:
                generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
        elif phase == 1:
            if helpers.EndsWithLeftDelimiter(generated):
                phase = 2
                if not helpers.CanConstrain(generated):
                    break
                elif helpers.IsComplete(generated):
                    generated, stepsLeft = helpers.AppendConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft)
                else:
                    generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
            else:
                generated, stepsLeft = helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft)
                if helpers.EndsWithLeftDelimiter(generated):
                    phase = 2
        else:
            if closed_spans == 0:
                if free_steps >= 24:
                    phase = 1
                    generated, stepsLeft = helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft)
                    if helpers.EndsWithLeftDelimiter(generated):
                        phase = 2
                elif free_steps >= 16 and not helpers.HasBudget(stepsLeft, 10):
                    phase = 1
                    generated, stepsLeft = helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft)
                    if helpers.EndsWithLeftDelimiter(generated):
                        phase = 2
                elif free_steps >= 12 and not helpers.HasBudget(stepsLeft, 6):
                    phase = 1
                    generated, stepsLeft = helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft)
                    if helpers.EndsWithLeftDelimiter(generated):
                        phase = 2
                elif free_steps >= 8 and not helpers.HasBudget(stepsLeft, 4):
                    phase = 1
                    generated, stepsLeft = helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft)
                    if helpers.EndsWithLeftDelimiter(generated):
                        phase = 2
                else:
                    generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
                    free_steps = free_steps + 1
            else:
                if post_span_free_steps >= 16:
                    phase = 1
                    generated, stepsLeft = helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft)
                    if helpers.EndsWithLeftDelimiter(generated):
                        phase = 2
                elif post_span_free_steps >= 8 and not helpers.HasBudget(stepsLeft, 8):
                    phase = 1
                    generated, stepsLeft = helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft)
                    if helpers.EndsWithLeftDelimiter(generated):
                        phase = 2
                elif post_span_free_steps >= 4 and not helpers.HasBudget(stepsLeft, 5):
                    phase = 1
                    generated, stepsLeft = helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft)
                    if helpers.EndsWithLeftDelimiter(generated):
                        phase = 2
                else:
                    generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
                    free_steps = free_steps + 1
                    post_span_free_steps = post_span_free_steps + 1
    remainingSteps = stepsLeft
    return generated, remainingSteps
