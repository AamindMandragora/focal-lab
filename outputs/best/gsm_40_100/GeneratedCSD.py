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
    # Strategy: delay the graded span even more aggressively and only open it after
    # a durable late-answer phase. The previous policy still allowed spans to begin
    # near intermediate arithmetic, which often caused the constrained expression to
    # capture a local quantity instead of the final answer. This version uses four
    # phases: extended free-form reasoning, a short wrap-up phase, an explicit
    # answer-cue phase, and then a persistent nudge phase that keeps encouraging the
    # natural left delimiter until it actually appears. This biases the model toward
    # finishing the whole verbal solution before entering the graded span.
    #
    # Inside the span, always use the positive guard
    # `helpers.IsComplete(generated) or helpers.CanConstrain(generated)` before
    # calling `helpers.AppendConstrainedOrRightDelimiterStep`, so a complete
    # expression can close with `>>` instead of exiting early. After the first
    # closed span, stop immediately to keep the final graded expression as the
    # terminal answer.
    # CSD_RATIONALE_END
    phase = "reason"
    inside_span = False
    closed_spans = 0
    reason_steps = 0
    wrap_steps = 0
    answer_steps = 0
    nudge_steps = 0

    # invariant helpers.lm == lm
    # invariant helpers.parser == parser
    # invariant lm.ValidTokensIdsLogits()
    # invariant 0 <= stepsLeft <= maxSteps
    # invariant |generated| + stepsLeft <= maxSteps
    # decreases stepsLeft
    while stepsLeft > 0:
        if inside_span:
            if helpers.IsComplete(generated) or helpers.CanConstrain(generated):
                generated, stepsLeft = helpers.AppendConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft)
                if helpers.EndsWithRightDelimiter(generated):
                    inside_span = False
                    closed_spans = closed_spans + 1
                    phase = "done"
            else:
                break
        elif closed_spans > 0:
            break
        elif phase == "reason":
            if reason_steps < 56 and helpers.HasBudget(stepsLeft, 12):
                generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
                reason_steps = reason_steps + 1
                if helpers.EndsWithLeftDelimiter(generated):
                    inside_span = True
                    phase = "span"
                elif reason_steps >= 56:
                    phase = "wrap"
                elif reason_steps >= 44 and not helpers.HasBudget(stepsLeft, 10):
                    phase = "wrap"
            else:
                phase = "wrap"
                if helpers.HasBudget(stepsLeft, 1):
                    generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
                    wrap_steps = wrap_steps + 1
                    if helpers.EndsWithLeftDelimiter(generated):
                        inside_span = True
                        phase = "span"
                else:
                    break
        elif phase == "wrap":
            if wrap_steps < 8 and helpers.HasBudget(stepsLeft, 8):
                generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
                wrap_steps = wrap_steps + 1
                if helpers.EndsWithLeftDelimiter(generated):
                    inside_span = True
                    phase = "span"
                elif wrap_steps >= 8:
                    phase = "answer"
                elif wrap_steps >= 4 and not helpers.HasBudget(stepsLeft, 7):
                    phase = "answer"
            else:
                phase = "answer"
                if helpers.HasBudget(stepsLeft, 1):
                    generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
                    answer_steps = answer_steps + 1
                    if helpers.EndsWithLeftDelimiter(generated):
                        inside_span = True
                        phase = "span"
                else:
                    break
        elif phase == "answer":
            if answer_steps < 10 and helpers.HasBudget(stepsLeft, 5):
                if answer_steps < 6:
                    generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
                else:
                    generated, stepsLeft = helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft)
                answer_steps = answer_steps + 1
                if helpers.EndsWithLeftDelimiter(generated):
                    inside_span = True
                    phase = "span"
                elif answer_steps >= 10:
                    phase = "nudge"
            else:
                phase = "nudge"
                if helpers.HasBudget(stepsLeft, 1):
                    generated, stepsLeft = helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft)
                    nudge_steps = nudge_steps + 1
                    if helpers.EndsWithLeftDelimiter(generated):
                        inside_span = True
                        phase = "span"
                else:
                    break
        elif phase == "nudge":
            if helpers.HasBudget(stepsLeft, 1):
                generated, stepsLeft = helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft)
                nudge_steps = nudge_steps + 1
                if helpers.EndsWithLeftDelimiter(generated):
                    inside_span = True
                    phase = "span"
            else:
                break
        elif phase == "span":
            if helpers.IsComplete(generated) or helpers.CanConstrain(generated):
                generated, stepsLeft = helpers.AppendConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft)
                if helpers.EndsWithRightDelimiter(generated):
                    inside_span = False
                    closed_spans = closed_spans + 1
                    phase = "done"
            else:
                break
        else:
            break
    remainingSteps = stepsLeft
    return generated, remainingSteps
