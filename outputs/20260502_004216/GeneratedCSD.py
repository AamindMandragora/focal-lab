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
    # - Generate ordinary reasoning with AppendUnconstrainedStep in natural mode.
    # - When budget gets tight or after any span has closed, switch to finalizing.
    # - In finalizing, continue unconstrained generation until a natural left
    #   delimiter appears; that event opens the verified answer span.
    # - Inside the span, use AppendConstrainedStep under the positive guard
    #   helpers.CanConstrain(generated).
    # - Treat EndsWithLeftDelimiter/EndsWithRightDelimiter as events and track
    #   persistent span state explicitly with in_span and closed_spans.
    # - Use a single bounded checkpoint for local recovery from dead states; after
    #   one restore, continue in finalizing mode without repeated rollback search.
    # - Stop only from explicit completion/dead-state branches, never from a bare
    #   top-level break in the loop body.
    # CSD_RATIONALE_END
    phase = "reason"
    in_span = False
    closed_spans = 0
    has_checkpoint = False
    checkpoint = []
    used_restore = False
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
        elif helpers.IsDead(generated):
            if has_checkpoint and not used_restore:
                generated = helpers.RestoreCheckpoint(checkpoint)
                used_restore = True
                has_checkpoint = False
                phase = "finalizing"
                in_span = False
            else:
                break
        elif in_span:
            if helpers.EndsWithRightDelimiter(generated):
                in_span = False
                closed_spans = closed_spans + 1
                if phase == "finalizing":
                    final_span_closed = True
                else:
                    phase = "reason"
            elif helpers.CanConstrain(generated):
                generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
                if helpers.EndsWithRightDelimiter(generated):
                    in_span = False
                    closed_spans = closed_spans + 1
                    if phase == "finalizing":
                        final_span_closed = True
                    else:
                        phase = "reason"
            else:
                break
        elif phase == "reason":
            if not has_checkpoint and not used_restore and stepsLeft >= 6:
                checkpoint = helpers.Checkpoint(generated)
                has_checkpoint = True

            if closed_spans > 0:
                phase = "finalizing"
                generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
                if helpers.EndsWithLeftDelimiter(generated):
                    in_span = True
            elif stepsLeft <= 4 or helpers.MinStepsToComplete(generated) >= stepsLeft:
                phase = "finalizing"
                generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
                if helpers.EndsWithLeftDelimiter(generated):
                    in_span = True
            else:
                generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
                if helpers.EndsWithLeftDelimiter(generated):
                    in_span = True
                    phase = "finalizing"
        elif phase == "finalizing":
            if not has_checkpoint and not used_restore:
                checkpoint = helpers.Checkpoint(generated)
                has_checkpoint = True

            if helpers.EndsWithLeftDelimiter(generated):
                in_span = True
            else:
                generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
                if helpers.EndsWithLeftDelimiter(generated):
                    in_span = True
        else:
            break

        if final_span_closed:
            break
        elif stepsLeft >= stepsLeftBeforeIteration:
            break
        if stepsLeft >= stepsLeftBeforeIteration:
            break
        else:
            break
    remainingSteps = stepsLeft
    return generated, remainingSteps
