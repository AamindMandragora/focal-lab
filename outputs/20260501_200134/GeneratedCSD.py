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
    # Generate natural text unconstrained until either budget is low or a left
    # delimiter is produced naturally. If a constrained span is entered, continue
    # with constrained steps while the current generated prefix is either complete
    # or constrainable. Once the constrained content is complete, close the span
    # explicitly with AppendRightDelimiter. After one closed span, stop; otherwise
    # continue unconstrained natural text. This preserves natural-delimiter style
    # while avoiding complex phase bookkeeping and recovery logic.
    # CSD_RATIONALE_END
    in_span = False
    closed_spans = 0

    # invariant helpers.lm == lm
    # invariant helpers.parser == parser
    # invariant lm.ValidTokensIdsLogits()
    # invariant 0 <= stepsLeft <= maxSteps
    # invariant |generated| + stepsLeft <= maxSteps
    # decreases stepsLeft
    while stepsLeft > 0:
        if in_span:
            if helpers.EndsWithRightDelimiter(generated):
                in_span = False
                closed_spans = closed_spans + 1
                break
            elif helpers.IsComplete(generated):
                generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
                if helpers.EndsWithRightDelimiter(generated):
                    in_span = False
                    closed_spans = closed_spans + 1
                    break
                else:
                    break
            elif helpers.CanConstrain(generated):
                generated, stepsLeft = helpers.AppendConstrainedStep([], generated, stepsLeft)
                if helpers.EndsWithRightDelimiter(generated):
                    in_span = False
                    closed_spans = closed_spans + 1
                    break
            else:
                break
        else:
            if closed_spans > 0:
                break
            elif stepsLeft <= 2:
                generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)
                if helpers.EndsWithLeftDelimiter(generated):
                    in_span = True
                else:
                    break
            else:
                generated, stepsLeft = helpers.AppendUnconstrainedStep([], generated, stepsLeft)
                if helpers.EndsWithLeftDelimiter(generated):
                    in_span = True
                elif helpers.EndsWithRightDelimiter(generated):
                    break
    remainingSteps = stepsLeft
    return generated, remainingSteps
