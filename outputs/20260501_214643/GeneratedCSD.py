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
    # Strategy: keep the control flow simple to aid verification.
    # We generate unconstrained text until a left delimiter appears naturally.
    # Once inside a constrained span, we emit constrained tokens until the span is
    # complete, then explicitly close it with the helper right-delimiter append.
    # After one closed constrained span, we stop. This preserves a natural-delimiter
    # style for opening while keeping delimiter closing explicit and verification
    # friendly.
    # CSD_RATIONALE_END
    inside_span = False
    closed_span = False

    # invariant helpers.lm == lm
    # invariant helpers.parser == parser
    # invariant lm.ValidTokensIdsLogits()
    # invariant 0 <= stepsLeft <= maxSteps
    # invariant |generated| + stepsLeft <= maxSteps
    # decreases stepsLeft
    while stepsLeft > 0:
        if closed_span:
            break
        elif inside_span:
            if helpers.IsComplete(generated):
                generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
                inside_span = False
                closed_span = True
            elif helpers.CanConstrain(generated):
                generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
            else:
                break
        else:
            generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
            if helpers.EndsWithLeftDelimiter(generated):
                inside_span = True
    remainingSteps = stepsLeft
    return generated, remainingSteps
