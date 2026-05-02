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
    # Strategy: generate ordinary text unconstrained until a natural left delimiter
    # appears or until budget pressure suggests explicitly opening one constrained
    # span. Once inside a span, keep emitting constrained tokens while the grammar
    # can still be constrained; as soon as the constrained content is complete,
    # close the span explicitly with the right delimiter. After one closed span,
    # stop. This keeps delimiter control explicit and uses only append helpers.
    # CSD_RATIONALE_END
    phase = "reason"
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
            if helpers.IsComplete(generated):
                generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
                in_span = False
                closed_spans = closed_spans + 1
                phase = "done"
            elif helpers.CanConstrain(generated):
                generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
                if helpers.EndsWithRightDelimiter(generated):
                    in_span = False
                    closed_spans = closed_spans + 1
                    phase = "done"
            else:
                break
        else:
            if phase == "done":
                break
            elif closed_spans > 0:
                break
            elif stepsLeft <= 2:
                generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)
                in_span = True
                phase = "finalize"
            else:
                generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
                if helpers.EndsWithLeftDelimiter(generated):
                    in_span = True
                    phase = "finalize"
                elif helpers.EndsWithRightDelimiter(generated):
                    closed_spans = closed_spans + 1
                    phase = "done"
    remainingSteps = stepsLeft
    return generated, remainingSteps
