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
    # Strategy: use natural-delimiter decoding with explicit span-state tracking.
    # Outside spans, advance only with unconstrained decoding. Once the decoding is
    # answer-ready, remain in unconstrained mode until a left delimiter is naturally
    # emitted. Inside a span, only call AppendConstrainedStep under an explicit
    # positive guard that checks helpers.CanConstrain(generated), while still
    # handling helpers.IsComplete(generated) first so completion can close the span
    # without treating temporary non-constrainable states as failure. Maintain one
    # bounded checkpoint per span for local recovery from dead continuations; after a
    # single recovery, fall back to ordinary unconstrained reasoning and wait for a
    # future natural left delimiter. Closed spans are counted explicitly, and the
    # final span closure terminates decoding.
    # CSD_RATIONALE_END
    phase = "reason"
    inside_span = False
    closed_spans = 0
    answer_ready = False
    checkpoint = []
    has_checkpoint = False
    final_span_closed = False
    recovered_this_span = False

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
        elif inside_span:
            if helpers.EndsWithRightDelimiter(generated):
                inside_span = False
                closed_spans = closed_spans + 1
                has_checkpoint = False
                recovered_this_span = False
                if phase == "final":
                    final_span_closed = True
                    break
                else:
                    phase = "reason"
            elif helpers.IsComplete(generated):
                if not helpers.EndsWithRightDelimiter(generated):
                    generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
                    if helpers.EndsWithRightDelimiter(generated):
                        inside_span = False
                        closed_spans = closed_spans + 1
                        has_checkpoint = False
                        recovered_this_span = False
                        if phase == "final":
                            final_span_closed = True
                            break
                        else:
                            phase = "reason"
                else:
                    inside_span = False
                    closed_spans = closed_spans + 1
                    has_checkpoint = False
                    recovered_this_span = False
                    if phase == "final":
                        final_span_closed = True
                        break
                    else:
                        phase = "reason"
            elif helpers.IsDead(generated):
                if has_checkpoint and not recovered_this_span:
                    generated = helpers.RestoreIfDead(generated, checkpoint)
                    inside_span = False
                    has_checkpoint = False
                    recovered_this_span = True
                    phase = "reason"
                    generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
                    if helpers.EndsWithLeftDelimiter(generated):
                        inside_span = True
                        has_checkpoint = False
                        recovered_this_span = False
                else:
                    inside_span = False
                    has_checkpoint = False
                    phase = "reason"
                    generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
                    if helpers.EndsWithLeftDelimiter(generated):
                        inside_span = True
                        has_checkpoint = False
                        recovered_this_span = False
            elif not has_checkpoint:
                if helpers.CanConstrain(generated):
                    checkpoint = helpers.Checkpoint(generated)
                    has_checkpoint = True
                    generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
                    if helpers.EndsWithRightDelimiter(generated):
                        inside_span = False
                        closed_spans = closed_spans + 1
                        has_checkpoint = False
                        recovered_this_span = False
                        if phase == "final":
                            final_span_closed = True
                            break
                        else:
                            phase = "reason"
                else:
                    inside_span = False
                    has_checkpoint = False
                    phase = "reason"
                    generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
                    if helpers.EndsWithLeftDelimiter(generated):
                        inside_span = True
                        has_checkpoint = False
                        recovered_this_span = False
            elif helpers.CanConstrain(generated):
                generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
                if helpers.EndsWithRightDelimiter(generated):
                    inside_span = False
                    closed_spans = closed_spans + 1
                    has_checkpoint = False
                    recovered_this_span = False
                    if phase == "final":
                        final_span_closed = True
                        break
                    else:
                        phase = "reason"
            else:
                inside_span = False
                has_checkpoint = False
                phase = "reason"
                generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
                if helpers.EndsWithLeftDelimiter(generated):
                    inside_span = True
                    has_checkpoint = False
                    recovered_this_span = False
        else:
            if closed_spans > 0:
                answer_ready = True
            elif stepsLeft <= 8:
                answer_ready = True
            elif helpers.MinStepsToComplete(generated) >= stepsLeft:
                answer_ready = True
            elif helpers.ParserDistanceToComplete(generated) >= stepsLeft:
                answer_ready = True
            elif helpers.ValidContinuationCount(generated) <= 1:
                answer_ready = True
            else:
                answer_ready = False

            if answer_ready:
                phase = "final"
            else:
                phase = "reason"

            generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
        if stepsLeft >= stepsLeftBeforeIteration:
            break
    remainingSteps = stepsLeft
    return generated, remainingSteps
