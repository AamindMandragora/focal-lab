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
    # Strategy: stay in ordinary unconstrained decoding outside answer spans and
    # track span state explicitly with `in_span`. Before the final answer span, keep
    # generating reasoning tokens unconstrained; once answer-ready, continue
    # unconstrained decoding until a natural left delimiter appears. Inside a span,
    # never call `AppendConstrainedStep` from a completeness branch. Instead:
    # - if the span is complete and already ends with a right delimiter, close it
    # - if the span is complete but not yet closed, take one unconstrained step so
    #   natural-delimiter mode can emit the right delimiter
    # - otherwise, if constrained continuation is available, take a constrained
    #   step.
    # Maintain only a bounded local checkpoint for complete span prefixes so dead
    # continuations can recover to the latest complete local state. After the final
    # answer span closes, terminate.
    # CSD_RATIONALE_END
    phase = "reason"
    in_span = False
    closed_spans = 0
    final_span_closed = False
    checkpoint = []
    has_checkpoint = False

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
                checkpoint = []
                has_checkpoint = False
                if phase == "answer":
                    final_span_closed = True
                    break
                phase = "reason"
                generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
            elif helpers.IsComplete(generated):
                checkpoint = helpers.Checkpoint(generated)
                has_checkpoint = True
                if helpers.ValidContinuationCount(generated) > 0:
                    generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
                    if helpers.EndsWithRightDelimiter(generated):
                        in_span = False
                        closed_spans = closed_spans + 1
                        checkpoint = []
                        has_checkpoint = False
                        if phase == "answer":
                            final_span_closed = True
                            break
                        phase = "reason"
                    else:
                        generated = helpers.RestoreIfDead(generated, checkpoint)
                else:
                    break
            elif helpers.CanConstrain(generated):
                generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
                if helpers.IsDead(generated):
                    if has_checkpoint:
                        generated = helpers.RestoreCheckpoint(checkpoint)
                        has_checkpoint = False
                        checkpoint = []
                    else:
                        break
            else:
                break
        else:
            if final_span_closed:
                break
            elif phase == "answer":
                generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
                if helpers.EndsWithLeftDelimiter(generated):
                    in_span = True
                    checkpoint = []
                    has_checkpoint = False
            elif closed_spans == 0 and helpers.MinStepsToComplete(generated) + 2 < stepsLeft:
                generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
                if helpers.EndsWithLeftDelimiter(generated):
                    in_span = True
                    checkpoint = []
                    has_checkpoint = False
                elif helpers.EndsWithRightDelimiter(generated):
                    closed_spans = closed_spans + 1
            else:
                phase = "answer"
                generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
                if helpers.EndsWithLeftDelimiter(generated):
                    in_span = True
                    checkpoint = []
                    has_checkpoint = False
    remainingSteps = stepsLeft
    return generated, remainingSteps
