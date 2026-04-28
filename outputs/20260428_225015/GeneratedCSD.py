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
    # Strategy: begin with ordinary free-form reasoning while delimiters are masked.
    # After enough free-form progress, an answer cue, or budget pressure, switch to
    # nudging the natural left delimiter until it appears. Once a left delimiter has
    # appeared, stay in constrained-span mode and use only constrained-or-right-
    # delimiter steps so a complete expression can close naturally. After the first
    # closed span, terminate.
    # CSD_RATIONALE_END
    closed_spans = 0
    freeform_steps = 0
    nudge_mode = False

    # invariant helpers.lm == lm
    # invariant helpers.parser == parser
    # invariant lm.ValidTokensIdsLogits()
    # invariant 0 <= stepsLeft <= maxSteps
    # invariant |generated| + stepsLeft <= maxSteps
    # decreases stepsLeft
    while stepsLeft > 0:
        if closed_spans > 0:
            break
        elif helpers.EndsWithRightDelimiter(generated):
            closed_spans += 1
            break
        elif helpers.EndsWithLeftDelimiter(generated):
            if helpers.IsComplete(generated):
                generated, stepsLeft = helpers.AppendConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft)
            elif helpers.CanConstrain(generated):
                generated, stepsLeft = helpers.AppendConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft)
            else:
                break
        else:
            min_needed = helpers.MinStepsToComplete(generated)
            distance = helpers.ParserDistanceToComplete(generated)
            valid_count = helpers.ValidContinuationCount(generated)

            needed = 3
            if min_needed > needed:
                needed = min_needed
            if distance > needed:
                needed = distance
            if valid_count == 1:
                needed = needed + 1

            budget_pressure = not helpers.HasBudget(stepsLeft, needed + 2)
            should_nudge = nudge_mode or freeform_steps >= 12 or budget_pressure

            if should_nudge:
                nudge_mode = True
                generated, stepsLeft = helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft)
            else:
                generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
                freeform_steps = freeform_steps + 1
    remainingSteps = stepsLeft
    return generated, remainingSteps
