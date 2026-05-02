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
    # For GSM, keep most decoding in ordinary unconstrained reasoning so the model
    # can finish multi-step arithmetic before committing to a verified span.
    # Delay answer opening until there is moderate evidence of wrap-up, avoiding the
    # first local calculation by requiring some reasoning progress and stronger
    # readiness signals before nudging toward `<<`.
    # Start nudging for a natural left delimiter early enough that multiple attempts
    # are possible, rather than waiting for a nearly exhausted budget.
    # Once a span is open, always prioritize the positive grammar guard
    # `helpers.IsComplete(generated) or helpers.CanConstrain(generated)` and use
    # constrained-or-close decoding so `>>` can be emitted exactly at completion.
    # Allow at most one earlier scratch span, but prefer a single final span by only
    # becoming answer-ready after sufficient reasoning progress or stronger
    # completion/budget cues.
    # CSD_RATIONALE_END
    phase = "reason"
    in_span = False
    closed_spans = 0
    reason_steps = 0
    nudge_steps = 0
    final_ready = False

    # invariant helpers.lm == lm
    # invariant helpers.parser == parser
    # invariant lm.ValidTokensIdsLogits()
    # invariant 0 <= stepsLeft <= maxSteps
    # invariant |generated| + stepsLeft <= maxSteps
    # decreases stepsLeft
    while stepsLeft > 0 and closed_spans < 2:
        stepsLeftBeforeIteration = stepsLeft
        if helpers.IsDead(generated):
            break
        elif in_span:
            if helpers.EndsWithRightDelimiter(generated):
                in_span = False
                closed_spans = closed_spans + 1
                if closed_spans >= 2:
                    phase = "done"
                    break
                else:
                    phase = "reason"
                    final_ready = True
                    nudge_steps = 0
            elif helpers.IsComplete(generated) or helpers.CanConstrain(generated):
                generated, stepsLeft = helpers.AppendConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft)
                if helpers.EndsWithRightDelimiter(generated):
                    in_span = False
                    closed_spans = closed_spans + 1
                    if closed_spans >= 2:
                        phase = "done"
                        break
                    else:
                        phase = "reason"
                        final_ready = True
                        nudge_steps = 0
            else:
                generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
        elif phase == "done" or closed_spans >= 2:
            break
        else:
            if helpers.EndsWithLeftDelimiter(generated):
                in_span = True
                phase = "span"
                if helpers.IsComplete(generated) or helpers.CanConstrain(generated):
                    generated, stepsLeft = helpers.AppendConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft)
                    if helpers.EndsWithRightDelimiter(generated):
                        in_span = False
                        closed_spans = closed_spans + 1
                        if closed_spans >= 2:
                            phase = "done"
                            break
                        else:
                            phase = "reason"
                            final_ready = True
                            nudge_steps = 0
                else:
                    generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
            else:
                near_complete = False
                tight_completion = False
                low_branching = False
                if helpers.IsComplete(generated) or helpers.CanConstrain(generated):
                    if helpers.ParserDistanceToComplete(generated) <= 4:
                        near_complete = True
                    if helpers.MinStepsToComplete(generated) <= 4:
                        tight_completion = True
                    if helpers.ValidContinuationCount(generated) <= 5:
                        low_branching = True

                enough_reasoning = reason_steps >= 10
                substantial_reasoning = reason_steps >= 6
                budget_window = stepsLeft <= 12
                budget_pressure = stepsLeft <= 8
                prior_span_exists = closed_spans >= 1

                if not final_ready:
                    if prior_span_exists:
                        final_ready = True
                    elif substantial_reasoning and (near_complete or tight_completion or low_branching):
                        final_ready = True
                    elif enough_reasoning and budget_window:
                        final_ready = True
                    elif reason_steps >= 14:
                        final_ready = True
                    elif budget_pressure and reason_steps >= 8:
                        final_ready = True

                if final_ready:
                    generated, stepsLeft = helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft)
                    nudge_steps = nudge_steps + 1
                    if helpers.EndsWithLeftDelimiter(generated):
                        in_span = True
                        phase = "span"
                    elif nudge_steps >= 4 and stepsLeft > 0:
                        generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
                else:
                    generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
                    reason_steps = reason_steps + 1
        if stepsLeft >= stepsLeftBeforeIteration:
            break
    remainingSteps = stepsLeft
    return generated, remainingSteps
