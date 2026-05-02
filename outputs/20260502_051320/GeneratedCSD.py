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
    # Strategy: use an explicit phase machine with three phases: free-form reasoning,
    # opening a natural delimiter span, and decoding inside that span. A real
    # closed-span counter is used in loop conditions so decoding can continue after
    # earlier scratch mini-expressions but stop after the final answer span has been
    # closed. In natural-delimiter mode, ordinary reasoning uses unconstrained
    # steps, opening uses repeated nudge-left-delimiter steps until a left delimiter
    # is actually emitted, and in-span decoding uses constrained-or-right-delimiter
    # steps guarded by the positive condition
    # (helpers.IsComplete(generated) or helpers.CanConstrain(generated)). Each loop
    # branch either consumes one helper step or breaks, and stepsLeft is updated
    # only from helper returns.
    # CSD_RATIONALE_END
    phase = "reason"
    closed_spans = 0

    # invariant helpers.lm == lm
    # invariant helpers.parser == parser
    # invariant lm.ValidTokensIdsLogits()
    # invariant 0 <= stepsLeft <= maxSteps
    # invariant |generated| + stepsLeft <= maxSteps
    # decreases stepsLeft
    while stepsLeft > 0 and closed_spans == 0:
        stepsLeftBeforeIteration = stepsLeft

        if phase == "reason":
            if helpers.EndsWithRightDelimiter(generated):
                closed_spans = closed_spans + 1
                if closed_spans > 0:
                    break
            elif helpers.EndsWithLeftDelimiter(generated):
                phase = "span"
                break
            elif helpers.IsComplete(generated):
                phase = "open"
                break
            elif helpers.MinStepsToComplete(generated) >= stepsLeft:
                phase = "open"
                break
            elif helpers.ParserDistanceToComplete(generated) + 1 >= stepsLeft:
                phase = "open"
                break
            elif helpers.ValidContinuationCount(generated) == 0:
                phase = "open"
                break
            else:
                generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
                if helpers.EndsWithRightDelimiter(generated):
                    closed_spans = closed_spans + 1
                elif helpers.EndsWithLeftDelimiter(generated):
                    phase = "span"

        elif phase == "open":
            if helpers.EndsWithRightDelimiter(generated):
                closed_spans = closed_spans + 1
                if closed_spans > 0:
                    break
            elif helpers.EndsWithLeftDelimiter(generated):
                phase = "span"
                break
            else:
                generated, stepsLeft = helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft)
                if helpers.EndsWithRightDelimiter(generated):
                    closed_spans = closed_spans + 1
                elif helpers.EndsWithLeftDelimiter(generated):
                    phase = "span"

        elif phase == "span":
            if helpers.EndsWithRightDelimiter(generated):
                closed_spans = closed_spans + 1
                if closed_spans > 0:
                    break
            elif helpers.IsComplete(generated) or helpers.CanConstrain(generated):
                generated, stepsLeft = helpers.AppendConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft)
                if helpers.EndsWithRightDelimiter(generated):
                    closed_spans = closed_spans + 1
                    if closed_spans > 0:
                        break
            else:
                break

        else:
            break

        if stepsLeft >= stepsLeftBeforeIteration:
            break
    remainingSteps = stepsLeft
    return generated, remainingSteps
