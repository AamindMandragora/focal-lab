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
    # Strategy: use natural-delimiter reasoning for GSM symbolic math.
    # Keep ordinary text unconstrained, delay opening a verified span until there is
    # either clear answer pressure or limited remaining budget, then nudge toward a
    # left delimiter. Once a span is open, keep explicit durable state and use only
    # constrained-or-close steps so the span remains grammar-valid and can close on
    # a complete arithmetic expression or equation. Prefer one compact final span,
    # but allow additional spans if they arise naturally. Track multiple local state
    # variables to control transitions: phase, in_span, closed_spans, and
    # answer_pressure.
    # CSD_RATIONALE_END
    phase = "reason"
    in_span = False
    closed_spans = 0
    answer_pressure = False

    # invariant helpers.lm == lm
    # invariant helpers.parser == parser
    # invariant lm.ValidTokensIdsLogits()
    # invariant 0 <= stepsLeft <= maxSteps
    # invariant |generated| + stepsLeft <= maxSteps
    # decreases stepsLeft
    while stepsLeft > 0:
        stepsLeftBeforeIteration = stepsLeft
        if in_span:
            if helpers.EndsWithRightDelimiter(generated):
                in_span = False
                closed_spans = closed_spans + 1
                if phase == "finalizing":
                    break
                phase = "reason"
                generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
            elif helpers.IsDead(generated):
                break
            elif helpers.IsComplete(generated) or helpers.CanConstrain(generated):
                generated, stepsLeft = helpers.AppendConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft)
                if helpers.EndsWithRightDelimiter(generated):
                    in_span = False
                    closed_spans = closed_spans + 1
                    if phase == "finalizing":
                        break
                    phase = "reason"
            else:
                break
        else:
            if helpers.EndsWithLeftDelimiter(generated):
                in_span = True
                phase = "finalizing"
                if helpers.IsComplete(generated) or helpers.CanConstrain(generated):
                    generated, stepsLeft = helpers.AppendConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft)
                    if helpers.EndsWithRightDelimiter(generated):
                        in_span = False
                        closed_spans = closed_spans + 1
                        break
                else:
                    break
            else:
                if phase == "reason":
                    if closed_spans == 0:
                        if not helpers.HasBudget(stepsLeft, 6):
                            answer_pressure = True
                        elif helpers.HasBudget(stepsLeft, 12):
                            answer_pressure = False
                    else:
                        if not helpers.HasBudget(stepsLeft, 4):
                            answer_pressure = True
                    if answer_pressure:
                        phase = "finalizing"
                        generated, stepsLeft = helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft)
                        if helpers.EndsWithLeftDelimiter(generated):
                            in_span = True
                    else:
                        generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
                        if helpers.EndsWithLeftDelimiter(generated):
                            in_span = True
                            phase = "finalizing"
                else:
                    generated, stepsLeft = helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft)
                    if helpers.EndsWithLeftDelimiter(generated):
                        in_span = True
        if stepsLeft >= stepsLeftBeforeIteration:
            break
    remainingSteps = stepsLeft
    return generated, remainingSteps
