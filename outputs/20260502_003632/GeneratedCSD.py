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
    # Strategy: natural-delimiter reasoning with explicit span-state and bounded
    # recovery. Generate ordinary reasoning unconstrained until final-answer
    # pressure, then continue unconstrained until a left delimiter is naturally
    # emitted. Once inside a span, use constrained decoding while the grammar is
    # complete-or-extendable, and close the span only when a right delimiter is
    # actually produced. Maintain a small checkpoint near completion for local
    # recovery from dead ends without making rollback the primary control flow.
    # Track `phase`, `inside_span`, and `closed_spans` explicitly because delimiter
    # suffix predicates are events, not persistent mode indicators.
    # CSD_RATIONALE_END
    phase = "reasoning"
    inside_span = False
    closed_spans = 0
    checkpoint = []
    has_checkpoint = False
    final_span_closed = False

    # invariant helpers.lm == lm
    # invariant helpers.parser == parser
    # invariant lm.ValidTokensIdsLogits()
    # invariant 0 <= stepsLeft <= maxSteps
    # invariant |generated| + stepsLeft <= maxSteps
    # decreases stepsLeft
    while stepsLeft > 0:
        stepsLeftBeforeIteration = stepsLeft
        if final_span_closed:
            break

        if inside_span:
            if helpers.EndsWithRightDelimiter(generated):
                inside_span = False
                closed_spans = closed_spans + 1
                has_checkpoint = False
                if phase == "finalizing":
                    final_span_closed = True
                else:
                    phase = "reasoning"
                break

            if helpers.IsDead(generated):
                if has_checkpoint:
                    generated = helpers.RestoreCheckpoint(checkpoint)
                    inside_span = False
                    has_checkpoint = False
                    phase = "finalizing"
                    break
                break

            if helpers.IsComplete(generated) and helpers.ValidContinuationCount(generated) <= 1:
                phase = "finalizing"

            if helpers.CanConstrain(generated):
                if (not has_checkpoint) and helpers.ParserDistanceToComplete(generated) <= 2:
                    checkpoint = helpers.Checkpoint(generated)
                    has_checkpoint = True
                generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
                if helpers.EndsWithRightDelimiter(generated):
                    inside_span = False
                    closed_spans = closed_spans + 1
                    has_checkpoint = False
                    if phase == "finalizing":
                        final_span_closed = True
                    else:
                        phase = "reasoning"
                break

            if helpers.IsComplete(generated):
                break

            if has_checkpoint:
                generated = helpers.RestoreCheckpoint(checkpoint)
                inside_span = False
                has_checkpoint = False
                phase = "finalizing"
                break
            break

        if helpers.EndsWithLeftDelimiter(generated):
            inside_span = True
            has_checkpoint = False
            if phase != "finalizing":
                phase = "scratch"
            break

        if helpers.EndsWithRightDelimiter(generated):
            closed_spans = closed_spans + 1
            if phase == "finalizing":
                final_span_closed = True
            else:
                phase = "reasoning"
            break

        if phase == "reasoning":
            if closed_spans > 0 or helpers.MinStepsToComplete(generated) >= stepsLeft:
                phase = "finalizing"
            generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
            if helpers.EndsWithLeftDelimiter(generated):
                inside_span = True
                has_checkpoint = False
                if phase != "finalizing":
                    phase = "scratch"
            elif helpers.EndsWithRightDelimiter(generated):
                closed_spans = closed_spans + 1
            break

        if phase == "scratch":
            if closed_spans >= 1 or helpers.MinStepsToComplete(generated) >= stepsLeft:
                phase = "finalizing"
            generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
            if helpers.EndsWithLeftDelimiter(generated):
                inside_span = True
                has_checkpoint = False
            elif helpers.EndsWithRightDelimiter(generated):
                closed_spans = closed_spans + 1
                phase = "reasoning"
            break

        if phase == "finalizing":
            generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
            if helpers.EndsWithLeftDelimiter(generated):
                inside_span = True
                has_checkpoint = False
            elif helpers.EndsWithRightDelimiter(generated):
                closed_spans = closed_spans + 1
                final_span_closed = True
            break

        break
        if stepsLeft >= stepsLeftBeforeIteration:
            break
        else:
            break
    remainingSteps = stepsLeft
    return generated, remainingSteps
