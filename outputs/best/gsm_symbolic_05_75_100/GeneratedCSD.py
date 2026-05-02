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
    # Strategy:
    # - Generate ordinary text unconstrained until a left delimiter is actually emitted.
    # - Track whether we are inside a constrained span only from delimiter endings.
    # - Inside a span, append constrained tokens only while the constrained content is
    #   not complete; once complete, explicitly close with AppendRightDelimiter.
    # - Outside a span, continue natural generation; if a right delimiter appears,
    #   treat the span as closed and, for a final span, stop.
    # - Avoid non-step state transitions inside the loop so every iteration either
    #   consumes one helper step or breaks, satisfying the decreases obligation.
    # - Avoid calling AppendConstrainedStep when helpers.IsComplete(generated) holds,
    #   since that helper requires the constrained suffix to be incomplete.
    # CSD_RATIONALE_END
    inside_span = False
    saw_closed_span = False
    final_mode = False

    # invariant helpers.lm == lm
    # invariant helpers.parser == parser
    # invariant lm.ValidTokensIdsLogits()
    # invariant 0 <= stepsLeft <= maxSteps
    # invariant |generated| + stepsLeft <= maxSteps
    # decreases stepsLeft
    while stepsLeft > 0:
        if inside_span:
            if helpers.EndsWithRightDelimiter(generated):
                inside_span = False
                saw_closed_span = True
                if final_mode:
                    break
                generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
                if helpers.EndsWithLeftDelimiter(generated):
                    inside_span = True
                elif helpers.EndsWithRightDelimiter(generated):
                    saw_closed_span = True
                continue
            if helpers.IsComplete(generated):
                generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
                inside_span = False
                saw_closed_span = True
                if final_mode:
                    break
                continue
            if helpers.CanConstrain(generated):
                generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
                if helpers.EndsWithRightDelimiter(generated):
                    inside_span = False
                    saw_closed_span = True
                continue
            break
        else:
            if helpers.EndsWithLeftDelimiter(generated):
                inside_span = True
                if helpers.IsComplete(generated):
                    generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
                    inside_span = False
                    saw_closed_span = True
                    if final_mode:
                        break
                elif helpers.CanConstrain(generated):
                    generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
                    if helpers.EndsWithRightDelimiter(generated):
                        inside_span = False
                        saw_closed_span = True
                else:
                    break
                continue
            if helpers.EndsWithRightDelimiter(generated):
                saw_closed_span = True
                if final_mode:
                    break
                generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
                if helpers.EndsWithLeftDelimiter(generated):
                    inside_span = True
                elif helpers.EndsWithRightDelimiter(generated):
                    saw_closed_span = True
                continue
            if not final_mode and saw_closed_span:
                final_mode = True
                generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
                if helpers.EndsWithLeftDelimiter(generated):
                    inside_span = True
                elif helpers.EndsWithRightDelimiter(generated):
                    saw_closed_span = True
                continue
            generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
            if helpers.EndsWithLeftDelimiter(generated):
                inside_span = True
            elif helpers.EndsWithRightDelimiter(generated):
                saw_closed_span = True
            continue
    remainingSteps = stepsLeft
    return generated, remainingSteps
