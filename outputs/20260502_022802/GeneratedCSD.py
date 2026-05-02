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
    # Strategy: generate ordinary reasoning in natural-delimiter mode, but track
    # both `closed_spans` and `scratch_spans` so loop/branch conditions distinguish
    # between an intermediate scratch mini-expression and the final answer span.
    # Continue decoding after a scratch span closes, but stop after the final span
    # closes. Outside spans, use ordinary unconstrained decoding until answer
    # pressure rises; then keep nudging toward a left delimiter until one is
    # emitted. Inside a span, use the positive guard
    # `helpers.CanConstrain(generated)` and emit
    # via `helpers.AppendConstrainedOrRightDelimiterStep(...)` so the span can
    # either extend valid content or close when complete. Use a bounded checkpoint
    # only for local recovery from dead states around an active span.
    # CSD_RATIONALE_END
    phase = "reason"
    in_span = False
    closed_spans = 0
    scratch_spans = 0
    nudge_mode = False
    has_checkpoint = False
    checkpoint = []

    # invariant helpers.lm == lm
    # invariant helpers.parser == parser
    # invariant lm.ValidTokensIdsLogits()
    # invariant 0 <= stepsLeft <= maxSteps
    # invariant |generated| + stepsLeft <= maxSteps
    # decreases stepsLeft
    while stepsLeft > 0 and (closed_spans == 0 or scratch_spans == closed_spans):
        stepsLeftBeforeIteration = stepsLeft
        if in_span:
            if helpers.IsDead(generated):
                if has_checkpoint:
                    generated = helpers.RestoreCheckpoint(checkpoint)
                    in_span = False
                    nudge_mode = True
                    phase = "final"
                    has_checkpoint = False
                    scratch_spans = closed_spans
                    break
                else:
                    break
            elif helpers.CanConstrain(generated):
                if not has_checkpoint:
                    checkpoint = helpers.Checkpoint(generated)
                    has_checkpoint = True
                generated, stepsLeft = helpers.AppendConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft)
                if helpers.EndsWithRightDelimiter(generated):
                    in_span = False
                    nudge_mode = False
                    closed_spans = closed_spans + 1
                    has_checkpoint = False
                    if phase == "scratch":
                        scratch_spans = closed_spans
                        phase = "reason"
                    else:
                        phase = "final_done"
                        break
                elif helpers.IsDead(generated):
                    if has_checkpoint:
                        generated = helpers.RestoreCheckpoint(checkpoint)
                        in_span = False
                        nudge_mode = True
                        phase = "final"
                        has_checkpoint = False
                        scratch_spans = closed_spans
                    else:
                        break
            else:
                break
        else:
            if helpers.IsDead(generated):
                if has_checkpoint:
                    generated = helpers.RestoreCheckpoint(checkpoint)
                    has_checkpoint = False
                    nudge_mode = True
                    phase = "final"
                    scratch_spans = closed_spans
                else:
                    break
            elif nudge_mode:
                generated, stepsLeft = helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft)
                if helpers.EndsWithLeftDelimiter(generated):
                    in_span = True
                    nudge_mode = False
                    checkpoint = helpers.Checkpoint(generated)
                    has_checkpoint = True
            else:
                need_final = closed_spans == 0
                budget_pressure = stepsLeft <= 4 + helpers.MinStepsToComplete(generated)
                near_complete = helpers.ParserDistanceToComplete(generated) <= 2
                many_choices = helpers.ValidContinuationCount(generated) > 1
                if need_final and (phase == "final" or budget_pressure or near_complete):
                    phase = "final"
                    nudge_mode = True
                    generated, stepsLeft = helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft)
                    if helpers.EndsWithLeftDelimiter(generated):
                        in_span = True
                        nudge_mode = False
                        checkpoint = helpers.Checkpoint(generated)
                        has_checkpoint = True
                elif closed_spans == 0 and not budget_pressure and not near_complete and many_choices:
                    generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
                    if helpers.EndsWithLeftDelimiter(generated):
                        in_span = True
                        nudge_mode = False
                        phase = "scratch"
                        checkpoint = helpers.Checkpoint(generated)
                        has_checkpoint = True
                else:
                    generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
                    if helpers.EndsWithLeftDelimiter(generated):
                        in_span = True
                        nudge_mode = False
                        if closed_spans == 0 and not budget_pressure and not near_complete:
                            phase = "scratch"
                        else:
                            phase = "final"
                        checkpoint = helpers.Checkpoint(generated)
                        has_checkpoint = True
        if stepsLeft >= stepsLeftBeforeIteration:
            break
    remainingSteps = stepsLeft
    return generated, remainingSteps
