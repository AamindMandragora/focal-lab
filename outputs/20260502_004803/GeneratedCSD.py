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
    # Strategy: use natural free-form reasoning outside delimiters and switch into a
    # durable answer/span mode when either budget pressure appears or a delimiter has
    # already been opened. Keep explicit state for whether we are inside a verified
    # span and how many spans have been closed. Outside spans, use only
    # AppendUnconstrainedStep. Inside spans, use the positive guard
    # IsComplete(...) or CanConstrain(...), then AppendConstrainedStep so the model
    # can either extend a grammar-valid arithmetic expression/equation or close the
    # delimiter naturally. Prefer delaying the final verified arithmetic span until
    # later, but allow multiple spans if they arise naturally. Use a checkpoint only
    # for bounded local recovery from dead parser states.
    # CSD_RATIONALE_END
    phase = "reason"
    inside_span = False
    closed_spans = 0
    answer_ready = False
    has_checkpoint = False
    checkpoint = []
    stall_count = 0

    # invariant helpers.lm == lm
    # invariant helpers.parser == parser
    # invariant lm.ValidTokensIdsLogits()
    # invariant 0 <= stepsLeft <= maxSteps
    # invariant |generated| + stepsLeft <= maxSteps
    # decreases stepsLeft
    while stepsLeft > 0:
        stepsLeftBeforeIteration = stepsLeft
        if inside_span:
            if helpers.EndsWithRightDelimiter(generated):
                inside_span = False
                closed_spans = closed_spans + 1
                phase = "post_span"
                has_checkpoint = False
                stall_count = 0
                if closed_spans >= 1 and not helpers.HasBudget(stepsLeft, 2):
                    break
            elif helpers.IsDead(generated):
                if has_checkpoint:
                    generated = helpers.RestoreIfDead(generated, checkpoint)
                    inside_span = False
                    phase = "reason"
                    has_checkpoint = False
                    stall_count = 0
                else:
                    break
            elif helpers.CanConstrain(generated):
                if not has_checkpoint and helpers.HasBudget(stepsLeft, 2):
                    checkpoint = helpers.Checkpoint(generated)
                    has_checkpoint = True
                prev_len = len(generated)
                generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
                if len(generated) == prev_len:
                    stall_count = stall_count + 1
                else:
                    stall_count = 0
                if helpers.EndsWithRightDelimiter(generated):
                    inside_span = False
                    closed_spans = closed_spans + 1
                    phase = "post_span"
                    has_checkpoint = False
                    stall_count = 0
                elif stall_count >= 2 and helpers.IsComplete(generated):
                    break
            else:
                break
        else:
            if helpers.EndsWithLeftDelimiter(generated):
                inside_span = True
                phase = "span"
                stall_count = 0
                if not has_checkpoint:
                    checkpoint = helpers.Checkpoint(generated)
                    has_checkpoint = True
            elif helpers.IsDead(generated):
                if has_checkpoint:
                    generated = helpers.RestoreIfDead(generated, checkpoint)
                    inside_span = False
                    phase = "reason"
                    has_checkpoint = False
                    stall_count = 0
                else:
                    break
            else:
                if closed_spans == 0:
                    if not helpers.HasBudget(stepsLeft, 6):
                        answer_ready = True
                    elif helpers.HasBudget(stepsLeft, 3) and helpers.MinStepsToComplete(generated) <= 2:
                        answer_ready = True
                else:
                    if not helpers.HasBudget(stepsLeft, 4):
                        break

                prev_len = len(generated)
                generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
                if len(generated) == prev_len:
                    stall_count = stall_count + 1
                else:
                    stall_count = 0

                if helpers.EndsWithLeftDelimiter(generated):
                    inside_span = True
                    phase = "span"
                    if not has_checkpoint:
                        checkpoint = helpers.Checkpoint(generated)
                        has_checkpoint = True
                    stall_count = 0
                elif answer_ready:
                    phase = "seek_span"
                else:
                    phase = "reason"

                if stall_count >= 2:
                    break
        if stepsLeft >= stepsLeftBeforeIteration:
            break
    remainingSteps = stepsLeft
    return generated, remainingSteps
