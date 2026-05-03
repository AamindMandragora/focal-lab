"""
Prompt templates for Python-first CSD strategy generation.

The model writes a strategy body for `generation/csd/GeneratedAgentTemplate.py`.
That body is later transpiled to Dafny for verification.
"""

import os
from functools import lru_cache
from pathlib import Path

from .strategy_memory import build_prompt_memory


PROMPTS_DIR = Path(__file__).resolve().parent
HELPER_REFERENCE_PATH = PROMPTS_DIR / "csd" / "VerifiedAgentSynthesis.md"

STANDARD_INVARIANT_BLOCK = """\
# invariant helpers.lm == lm
# invariant helpers.parser == parser
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# invariant |generated| + stepsLeft <= maxSteps
# decreases stepsLeft
"""

REMOVED_HELPER_GUIDANCE = """\
Do not use removed helpers or stale legacy patterns from older prompt drafts:
- no soft constrained steps as the main answer-channel policy
- no top-k constrained experimentation as the default policy
- no budget-aware switching as the main outer phase policy
Keep the strategy on the curated helper surface instead.
"""

CURATED_HELPER_REFERENCE = """\
# Curated Helper Mini-Reference

Use these helpers exactly as named. Do not invent helper names.

## Core State

- `generated` is `list[str]`, not a string.
- All emitted tokens go into `generated`.
- `prompt` is read-only.
- The final graded answer must be emitted inside a grammar-valid delimiter span.
- Split-prefix strategies are allowed when you intentionally track a stable prefix plus the
  current constrained span, but single-prefix control is still the default.

## Proof-Carrying Loop Pattern

Place these lines immediately above each decoding `while` loop:

```python
# invariant helpers.lm == lm
# invariant helpers.parser == parser
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# invariant |generated| + stepsLeft <= maxSteps
# decreases stepsLeft
```

Every live branch inside that loop must either consume a helper step or `break`.

## Natural Delimiters For GSM

- Ordinary reasoning should use `helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)`.
- Answer-opening pressure should use
  `helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft)`.
- Track span openings with `helpers.EndsWithLeftDelimiter(generated)`.
- Track span closures with `helpers.EndsWithRightDelimiter(generated)`.
- Inside a natural span, prefer
  `helpers.AppendConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft)`.
- Use a positive inside-span guard such as
  `helpers.IsComplete(generated) or helpers.CanConstrain(generated)`.
- Do not break on `not helpers.CanConstrain(generated)` before checking completion.

## Explicit Delimiters For Non-Natural Runs

- Emit delimiters with visible executable calls:
  `generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)`
  `generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)`
- Emit constrained content with:
  `generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)`
- Emit the left delimiter before constrained tokens.
- Emit the right delimiter only after the answer segment is complete.

## Grammar / Span Wrappers

- `helpers.LongestValidSuffix(generated)`
- `helpers.CanConstrain(generated)`
- `helpers.IsComplete(generated)`
- `helpers.IsDead(generated)`
- `helpers.ValidContinuationCount(generated)`
- `helpers.ParserDistanceToComplete(generated)`
- `helpers.MinStepsToComplete(generated)`
- `helpers.EndsWithLeftDelimiter(generated)`
- `helpers.EndsWithRightDelimiter(generated)`
- `helpers.LastTokenBefore(generated, target)`
- `helpers.CountOccurrences(generated, target)`
- `helpers.TokensSinceLastDelimiter(generated)`

## Split-Prefix / Arithmetic-Biased Policies

- Split-prefix GSM policies are allowed when you intentionally track durable local state such as
  `inside_constrained` / `insideConstrainedOut` and `current_constrained`.
- The supported helper family includes:
  `helpers.OpenConstrainedSpan(...)`,
  `helpers.AppendConstrainedToken(...)`,
  `helpers.CloseConstrainedSpan(...)`,
  `helpers.AdaptiveConstrainedStep(...)`,
  `helpers.GroupBoostedConstrainedStep(...)`,
  and `helpers.PenalizedConstrainedStep(...)`.
- For split-prefix policies, direct parser calls on the tracked constrained suffix are allowed, e.g.
  `parser.IsCompletePrefix(current_constrained)`.
- Arithmetic-aware opening cues such as `helpers.LastTokenBefore(generated, ">>")`,
  `helpers.LastTokenBefore(generated, "=")`, and `helpers.CountOccurrences(...)` are valid control
  signals when they drive a real helper step in the same branch.
- If you use `AdaptiveConstrainedStep`, it is fine to keep a small local
  `valid_token_groups` / `validTokenGroups` list and a `narrow_threshold` / `narrowThreshold`
  parameter for arithmetic token biasing. Those are not the same thing as a tiny phase quota.

## Checkpoints And Recovery

- `helpers.Checkpoint(generated)`
- `helpers.RestoreCheckpoint(checkpoint)`
- `helpers.RestoreIfDead(generated, checkpoint)`

Checkpoint helpers are available, but they are verifier-heavier than the direct
one-step-per-iteration policies preferred for GSM natural-delimiter runs.
"""

NATURAL_DELIMITER_OVERRIDE = """\
## GSM Natural-Delimiter Policy Override

This run enables natural GSM delimiter control. These rules override generic explicit-delimiter
advice for GSM only.

- Do not use `AppendLeftDelimiter`, `AppendRightDelimiter`, `AppendForcedToken`, or
  `ForcedTokenStep` for GSM delimiters.
- Use delimiter-masked `AppendUnconstrainedStep(...)` for ordinary free-form reasoning before the
  strategy is ready to open a span.
- Once the strategy is answer-ready, persistently use
  `AppendUnconstrainedNudgeLeftDelimiterStep(...)` until `helpers.EndsWithLeftDelimiter(generated)`
  becomes true.
- Inside the span, use `AppendConstrainedOrRightDelimiterStep(...)` and keep explicit open-span
  state until `helpers.EndsWithRightDelimiter(generated)` becomes true.
- Track a real closed-span counter such as `closed_spans`; do not use token counters as a span
  counter.
- Do not rely on tiny fixed quotas such as 4, 6, 8, 10, or 12 steps as the main finality policy.
- If a branch changes state into answer-opening or in-span mode, that same branch must also
  consume a helper step or `break`.
"""

SYSTEM_PROMPT = """\
You are generating the BODY of a Python function for a verified constrained
decoding strategy. Output only Python statements for the body: no imports, no
function signature, and no markdown fences.

The surrounding template already defines:

  helpers = CSDHelpers(lm, parser)
  generated = []
  stepsLeft = maxSteps

Do not redeclare those names and do not return manually.

Your body must contain:
- a `# CSD_RATIONALE_BEGIN` / `# CSD_RATIONALE_END` block
- a `# CSD_PROOF_SKETCH_BEGIN` / `# CSD_PROOF_SKETCH_END` block
- executable decoding code after those blocks

The proof sketch should explain parser validity, delimiter discipline, helper
preconditions, budget accounting, and termination/progress.

Every decoding `while` loop must have this exact comment block immediately above it:

```python
# invariant helpers.lm == lm
# invariant helpers.parser == parser
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# invariant |generated| + stepsLeft <= maxSteps
# decreases stepsLeft
```

Inside loops, every branch must either consume a helper step or `break`.
Never manually mutate `stepsLeft`; only update it from helper returns.
Use only the curated helper surface. Prefer helper wrappers such as
`helpers.IsComplete(generated)`, `helpers.ValidContinuationCount(generated)`,
`helpers.ParserDistanceToComplete(generated)`, and
`helpers.MinStepsToComplete(generated)` over direct parser calls on `generated`.
Direct parser calls on a tracked split-prefix suffix such as `current_constrained`
are allowed when using `OpenConstrainedSpan` / `AdaptiveConstrainedStep`.
"""

INITIAL_GENERATION_PROMPT = """\
Generate a Python strategy body for this use-case:

Use-case description: {task_description}

Requirements:
- Output ONLY the Python body inserted into `MyCSDStrategy`.
- Start with the required rationale block.
- Immediately after it, include the required proof-sketch block.
- Use a budget-bounded `while` loop with the exact invariant block from the system prompt.
- Use only the curated helper surface. Prefer append-style helpers.
- In natural GSM mode, ordinary reasoning should use `helpers.AppendUnconstrainedStep(...)`.
- In natural GSM mode, answer-opening pressure should use
  `helpers.AppendUnconstrainedNudgeLeftDelimiterStep(...)`.
- Track span boundaries with `helpers.EndsWithLeftDelimiter(generated)` and
  `helpers.EndsWithRightDelimiter(generated)`.
- Inside a natural span, prefer
  `helpers.AppendConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft)`.
- For non-natural runs, emit visible delimiter calls and use
  `helpers.AppendConstrainedStep(prompt, generated, stepsLeft)` for constrained tokens.
- Checkpoint helpers remain available when truly needed:
  `helpers.Checkpoint(generated)` and `helpers.RestoreIfDead(generated, checkpoint)`.
- Do not manually change `stepsLeft`; no `stepsLeft -= 1`, no `stepsLeft += 1`, and no
  `stepsLeft = stepsLeft - ...`. Helper calls already consume budget.
- Keep all parser-handled content inside grammar-valid delimiter spans.
- If a branch changes state into answer-opening or span mode, that same branch must also consume
  a helper step or `break`.
- Keep at least two meaningful local state variables that affect control flow.
- Avoid checkpoint rollback by default; it is verifier-heavier than direct step-by-step control.
- Prefer helper wrappers over direct parser calls on `generated`.
- A split-prefix GSM policy is also allowed: keep durable local state such as
  `inside_constrained` and `current_constrained`, use
  `OpenConstrainedSpan` / `AppendConstrainedToken` / `CloseConstrainedSpan`,
  and optionally use `AdaptiveConstrainedStep` with a small local arithmetic token-group bias.
- In split-prefix mode, direct parser calls on `current_constrained` are allowed, e.g.
  `parser.IsCompletePrefix(current_constrained)`.
- In natural delimiter mode, do not use `AppendLeftDelimiter`, `AppendRightDelimiter`,
  `AppendForcedToken`, or `ForcedTokenStep` in the executable body.
- Start natural answer opening early enough to have several nudge attempts; do not wait for
  thresholds like `stepsLeft <= 4` or `not helpers.HasBudget(stepsLeft, 6)`.
- Do not rely on tiny fixed phase quotas such as 4, 6, 8, 10, or 12 setup, wrap, or answer steps
  as the main finality policy.
- A good inside-span guard is:
  `helpers.IsComplete(generated) or helpers.CanConstrain(generated)`.
- Two good GSM directions are:
  1. a durable late-open final-span policy with long reasoning and persistent nudging for a
     single final span
  2. a scratch-to-final policy where raw unconstrained steps observe equation cues and later open
     reusable scratch spans before a final composing span
  3. a split-prefix arithmetic-biased policy where outside-span reasoning watches arithmetic cues,
     opening uses `OpenConstrainedSpan`, inside-span decoding uses
     `AdaptiveConstrainedStep` plus `AppendConstrainedToken`, and completion uses
     `CloseConstrainedSpan`

{removed_helper_guidance}
"""

VERIFICATION_ERROR_REFINEMENT_PROMPT = """\
The previous strategy failed verification or transpilation.

Previous strategy:
{previous_strategy}

Error:
{error_message}

{structured_feedback_block}{error_history_block}{behavioral_context_block}Rewrite the strategy body using only the curated helper surface.
Preserve the natural-delimiter style when applicable.
Prefer helper wrappers such as `helpers.IsComplete(generated)` over direct parser calls.
If a top-k call appears, replace broad top-k behavior with `AppendConstrainedStep(...)` or literal
`k = 1`.
If `decreases stepsLeft` may not decrease, make every live branch consume a helper step or `break`.
Use the structured verifier details to target the actual failed obligation rather than making a broad rewrite.
Do not use removed helpers or stale legacy patterns.
Return only the corrected Python body with the required rationale and proof-sketch blocks.
"""

RUNTIME_ERROR_REFINEMENT_PROMPT = """\
The previous strategy failed at runtime.

Previous strategy:
{previous_strategy}

Traceback:
{error_traceback}

Rewrite the strategy body using only the curated helper surface.
Assign every Append* helper result back into `generated, stepsLeft`; raw token-returning steps must
append `next_token` and set `stepsLeft = new_steps`.
Return only the corrected Python body with the required rationale and proof-sketch blocks.
"""

EVALUATION_FAILURE_REFINEMENT_PROMPT = """\
The previous strategy verified and ran, but evaluation was below threshold.

Previous strategy:
{previous_strategy}

Evaluation feedback:
{evaluation_feedback}

Improve the control policy using the current natural-delimiter helpers.
For GSM, prefer ordinary reasoning with `AppendUnconstrainedStep`, then natural answer opening
with `AppendUnconstrainedNudgeLeftDelimiterStep`, then
`AppendConstrainedOrRightDelimiterStep` while
`helpers.IsComplete(generated) or helpers.CanConstrain(generated)`.
If the strategy is using split-prefix helpers (`OpenConstrainedSpan`, `AdaptiveConstrainedStep`,
`AppendConstrainedToken`, `CloseConstrainedSpan`) and that family is close, keep it and repair
the arithmetic-cue or finality logic rather than flattening it into a generic append-only loop.
Avoid opening on the first local calculation, but also do not wait until only a tiny budget
remains before nudging for `<<`; natural delimiter opening needs several attempts.
Never put a `not helpers.CanConstrain(generated): break` branch before the completion-aware close
branch; completion is when `>>` becomes allowed, not when the strategy should exit.
If the current idea is only narrowly failing, keep the good parts and make a local repair.
If the metrics are fundamentally bad, rethink the policy instead of making a tiny edit.
Return only the corrected Python body with the required rationale and proof-sketch blocks.
"""

FORMAT_REPAIR_PROMPT = """\
Your output must be a Python method body. It is missing one or both required reasoning blocks.

Previous strategy:
{previous_strategy}

Rewrite it with:
# CSD_RATIONALE_BEGIN
# concise rationale
# CSD_RATIONALE_END
# CSD_PROOF_SKETCH_BEGIN
# concise sketch of parser-validity, delimiter, budget, and termination proof
# CSD_PROOF_SKETCH_END

Then include the executable strategy body. Keep helper names on the supported surface.
Return only Python statements.
"""

STRUCTURE_REPAIR_PROMPT = """\
The previous strategy has a structural issue:
{issue}

Previous strategy:
{previous_strategy}

Rewrite the body using only the curated helper surface.
Prefer:
- `AppendUnconstrainedStep`
- `AppendConstrainedStep`
- `AppendSoftConstrainedStep` only when explicitly guarded and truly necessary
- `AppendUnconstrainedNudgeLeftDelimiterStep`
- `AppendConstrainedOrRightDelimiterStep`
- `EndsWithLeftDelimiter` / `EndsWithRightDelimiter`
- `IsComplete`, `CanConstrain`, `ValidContinuationCount`,
  `ParserDistanceToComplete`, `MinStepsToComplete`
- `LastTokenBefore`, `CountOccurrences`, `TokensSinceLastDelimiter`

If helpful, you may also use split-prefix helpers such as
`OpenConstrainedSpan`, `AppendConstrainedToken`, `CloseConstrainedSpan`,
`AdaptiveConstrainedStep`, `GroupBoostedConstrainedStep`, and
`PenalizedConstrainedStep`.

Do not use removed helpers or stale legacy patterns.
Avoid checkpoint rollback unless the task-specific prompt explicitly requires it.
Return only the corrected Python body with the required rationale and proof-sketch blocks.
"""

DAFNY_SYSTEM_PROMPT = """\
You are generating the BODY of a Dafny method for a verified constrained
decoding strategy. Output only Dafny statements for the method body: no module
header, no method signature, and no markdown fences.

The surrounding Dafny template already defines:

  var helpers := new CSDHelpers(lm, parser);
  lm.ValidTokensIdsLogitsAlways();
  generated := [];
  var stepsLeft := maxSteps;

Do not redeclare those names and do not write the surrounding module.

Start with:
// CSD_RATIONALE_BEGIN
// ...
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// ...
// CSD_PROOF_SKETCH_END

Then write executable Dafny statements.

Every decoding loop must look like:

while stepsLeft > 0
  invariant helpers.lm == lm
  invariant helpers.parser == parser
  invariant lm.ValidTokensIdsLogits()
  invariant 0 <= stepsLeft <= maxSteps
  invariant |generated| + stepsLeft <= maxSteps
  decreases stepsLeft
{
  ...
}

Inside the loop, every non-break branch must consume a helper step.
Never manually decrement stepsLeft.
"""

DAFNY_INITIAL_GENERATION_PROMPT = """\
Generate a Dafny strategy body for this use-case:

Use-case description: {task_description}

Requirements:
- Output ONLY the Dafny method body inserted into `MyCSDStrategy`.
- Use Dafny syntax for assignments (`:=`), booleans (`true` / `false`), and loop invariants.
- Keep the rationale and proof-sketch blocks at the top using `//` comments.
- For GSM natural-delimiter mode, use `helpers.AppendUnconstrainedStep(...)` for ordinary
  reasoning, `helpers.AppendUnconstrainedNudgeLeftDelimiterStep(...)` when answer-opening
  pressure begins, and `helpers.AppendConstrainedOrRightDelimiterStep(...)` inside the span.
- A split-prefix GSM policy is also allowed when it is the main idea: use durable
  `inside_constrained` / `current_constrained` state, open with `OpenConstrainedSpan(...)`,
  decode with `AdaptiveConstrainedStep(...)` plus `AppendConstrainedToken(...)`, and close with
  `CloseConstrainedSpan(...)`.
- In split-prefix mode, direct parser calls on `current_constrained` are allowed.
- Do not use `AppendLeftDelimiter`, `AppendRightDelimiter`, `AppendForcedToken`, or
  `ForcedTokenStep` in GSM natural mode.
- Keep explicit state such as `phase`, `inside_span`, and `closed_spans`.
- Do not create state-only transitions in the loop. If a branch changes into open/span/final
  state, that same branch must also consume a helper step or break.
- Avoid tiny fixed phase quotas such as 4, 6, 8, 10, or 12 as the main finality policy.
- Prefer one of these shapes:
  1. a durable late-open final-span policy with long reasoning before persistent nudging
  2. a scratch-to-final policy where raw unconstrained steps observe arithmetic cues and later
     open reusable scratch spans before a final composing span
  3. a split-prefix arithmetic-biased policy where arithmetic cues or recent-span context trigger
     `OpenConstrainedSpan`, constrained decoding uses `AdaptiveConstrainedStep`, and completion
     closes with `CloseConstrainedSpan`
"""

DAFNY_VERIFICATION_ERROR_REFINEMENT_PROMPT = """\
The previous Dafny strategy body failed verification or compilation.

Previous strategy:
{previous_strategy}

Error:
{error_message}

Rewrite the Dafny method body. Keep the rationale and proof-sketch blocks, use Dafny syntax, and
return only the corrected body.
"""

DAFNY_RUNTIME_ERROR_REFINEMENT_PROMPT = """\
The previous Dafny strategy verified and compiled, but the compiled Python artifact failed at runtime.

Previous strategy:
{previous_strategy}

Traceback:
{error_traceback}

Rewrite the Dafny body to preserve the intended control policy while fixing the runtime issue.
Return only the corrected Dafny body.
"""

DAFNY_EVALUATION_FAILURE_REFINEMENT_PROMPT = """\
The previous Dafny strategy verified, compiled, and ran, but evaluation was below threshold.

Previous strategy:
{previous_strategy}

Evaluation feedback:
{evaluation_feedback}

Improve the Dafny control policy. If the current idea is close, keep the working structure and
make a small repair; if it is fundamentally wrong, redesign it instead of making a tiny edit.
Split-prefix GSM policies using `OpenConstrainedSpan`, `AdaptiveConstrainedStep`,
`AppendConstrainedToken`, and `CloseConstrainedSpan` are allowed when that structure is the
right fit for the failure feedback.
Return only the corrected Dafny body.
"""

DAFNY_FORMAT_REPAIR_PROMPT = """\
Your output must be a Dafny method body.

Previous strategy:
{previous_strategy}

Rewrite it with:
// CSD_RATIONALE_BEGIN
// concise rationale
// CSD_RATIONALE_END
// CSD_PROOF_SKETCH_BEGIN
// concise proof sketch
// CSD_PROOF_SKETCH_END

Then include the executable Dafny body. Return only Dafny statements.
"""

DAFNY_STRUCTURE_REPAIR_PROMPT = """\
The previous Dafny strategy has a structural issue:
{issue}

Previous strategy:
{previous_strategy}

Rewrite the Dafny method body to satisfy the issue. Keep the rationale and proof-sketch blocks,
use a budget-bounded `while stepsLeft > 0` loop with explicit invariants, and ensure every live
branch consumes a helper step or breaks.
"""


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip() in {"1", "true", "True", "yes", "on"}


def helper_reference_mode() -> str:
    explicit = os.environ.get("CSD_HELPER_REFERENCE_MODE", "").strip().lower()
    if explicit in {"curated", "mini", "compact"}:
        return "curated"
    if explicit in {"full", "markdown", "md"}:
        return "full"
    if explicit in {"none", "off", "0", "false"}:
        return "none"
    if _env_flag("CSD_INCLUDE_HELPER_REFERENCE_MD"):
        return "full"
    return "none"


@lru_cache(maxsize=1)
def _load_helper_reference_markdown() -> str:
    return HELPER_REFERENCE_PATH.read_text(encoding="utf-8").strip()


def _strip_example_sections(markdown_text: str) -> str:
    """
    Remove example-centric sections from the full helper markdown before prompt injection.
    """
    lines = markdown_text.splitlines()
    out: list[str] = []
    skipping = False
    skip_level = 0

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            title = stripped[level:].strip().lower()
            if skipping and level <= skip_level:
                skipping = False
            if any(marker in title for marker in ("example", "skeleton")):
                skipping = True
                skip_level = level
                continue
        if not skipping:
            out.append(line)

    return "\n".join(out).strip()


def _compose_system_prompt() -> str:
    mode = helper_reference_mode()
    natural_override = ""
    if _env_flag("CSD_REQUIRE_NATURAL_DELIMITERS"):
        natural_override = "\n\n" + NATURAL_DELIMITER_OVERRIDE.strip()

    if mode == "none":
        return SYSTEM_PROMPT + natural_override

    if mode == "curated":
        return (
            SYSTEM_PROMPT
            + "\n\n## Additional Curated Helper Reference\n\n"
            + "The following mini-reference is distilled from `generation/csd/VerifiedAgentSynthesis.md`.\n"
            + "Use it as the authoritative short reference for helper names, proof-critical usage,\n"
            + "checkpoint availability, and delimiter discipline.\n\n"
            + "[BEGIN CURATED_HELPER_REFERENCE]\n"
            + CURATED_HELPER_REFERENCE.strip()
            + "\n[END CURATED_HELPER_REFERENCE]\n"
            + natural_override
        )

    reference = _strip_example_sections(_load_helper_reference_markdown())
    return (
        SYSTEM_PROMPT
        + "\n\n## Additional Authoritative Helper Reference\n\n"
        + "Verified Agent Synthesis Helper Surface\n"
        + "The following markdown is copied from `generation/csd/VerifiedAgentSynthesis.md` with example sections removed.\n"
        + "Use it as the authoritative reference for helper names, signatures, object ownership, and contracts.\n"
        + "If this reference conflicts with your memory, follow the reference.\n\n"
        + "[BEGIN VERIFIED_AGENT_SYNTHESIS_MD]\n"
        + reference
        + "\n[END VERIFIED_AGENT_SYNTHESIS_MD]\n"
        + natural_override
    )


def _compose_dafny_system_prompt() -> str:
    natural_override = ""
    if _env_flag("CSD_REQUIRE_NATURAL_DELIMITERS"):
        natural_override = "\n\n" + NATURAL_DELIMITER_OVERRIDE.strip()
    return DAFNY_SYSTEM_PROMPT + natural_override


def _natural_delimiter_user_reminder() -> str:
    if not _env_flag("CSD_REQUIRE_NATURAL_DELIMITERS"):
        return ""
    return """\

Natural-delimiter mode reminder:
- Do not use `AppendLeftDelimiter`, `AppendRightDelimiter`,
  `AppendForcedToken`, or `ForcedTokenStep` in GSM natural mode.
- Use `AppendUnconstrainedStep` for ordinary reasoning, or raw `UnconstrainedStep` when the policy
  needs to observe the emitted token and set local state.
- The scanning helpers `LastTokenBefore`, `CountOccurrences`, and
  `TokensSinceLastDelimiter` are available for those raw-token policies.
- Split-prefix arithmetic policies are also allowed when they materially help GSM:
  keep local `inside_constrained` / `current_constrained` state, open with
  `OpenConstrainedSpan(...)`, decode with `AdaptiveConstrainedStep(...)` plus
  `AppendConstrainedToken(...)`, and close with `CloseConstrainedSpan(...)`.
- In those split-prefix policies, direct parser calls on `current_constrained` are allowed.
- Once answer-ready, use `AppendUnconstrainedNudgeLeftDelimiterStep` until
  `helpers.EndsWithLeftDelimiter(generated)` is true.
- Inside the span, use `AppendConstrainedOrRightDelimiterStep` until
  `helpers.EndsWithRightDelimiter(generated)` is true.
- Do not use plain `AppendConstrainedStep` in natural mode, and do not break on
  `not helpers.CanConstrain(generated)` before checking `helpers.IsComplete`.
- Prefer the positive span guard
  `helpers.IsComplete(generated) or helpers.CanConstrain(generated)`, followed
  by `AppendConstrainedOrRightDelimiterStep`.
- Track open-span state explicitly (`phase`, `inside_span`, or `in_span`);
  `EndsWithLeftDelimiter` is an opening event, not a persistent span-mode
  predicate.
- Track closed spans with a real counter such as `closed_spans`.
- In a `# decreases stepsLeft` loop, do not change into open/span/final-ready
  state without also consuming a helper step in that same branch.
- If the strategy deliberately uses scratch spans, an observed `=` / ` =` token after a quantity
  name is a good arithmetic-opening cue: set scratch-opening intent, then nudge for `<<`, constrain
  the arithmetic span, close it, and continue unless final state says the span is the answer.
"""


def _scratch_span_preference_reminder() -> str:
    if not _env_flag("CSD_GSM_PREFER_SCRATCH_SPANS"):
        return ""
    return """\

Scratch-span preference reminder (GSM):
- Earlier non-final spans should be named scratch assignments like
  `<<x_1 = ...>>`, not anonymous local fragments.
- The final span should compose scratch values with remaining constants, e.g.
  `<<x_1 + x_2 + 13>>`.
- Ported GSM v15 idea: use raw delimiter-masked reasoning steps when useful so the strategy can
  observe equation cues. If `next_token == "="` or `next_token == " ="`, treat the following tokens
  as arithmetic-worthy and begin persistent natural left-delimiter nudging for a scratch span.
- Do not stop after the first closed span unless it is explicitly final-ready.
- After closing a scratch span, continue free-form reasoning and later open
  another verified span.
- Keep this natural and adaptive: avoid rigid templates or fixed span counts.
"""


def _spider_single_sql_span_reminder() -> str:
    if not _env_flag("CSD_SPIDER_FORCE_SINGLE_SQL_SPAN"):
        return ""
    return """\

Spider SQL span reminder:
- Prefer explicit SQL-span opening for Spider with `helpers.AppendLeftDelimiter(...)`.
- In Spider start-at-span mode, open with `AppendLeftDelimiter(...)` before any
  unconstrained token generation. This mirrors the best Spider policy: spend the token budget on
  parser-governed SQL, not on free-form narration.
- Inside the SQL span, prefer `AppendConstrainedOrRightDelimiterStep` so completion can close
  `>>` naturally; avoid constrained-only loops that never emit a right delimiter.
- Do not broadly boost or hand-shape many token groups from the strategy. Let parser masking drive
  SQL syntax and keep any steering lightweight.
- Avoid rollback-heavy SQL policies by default. The best direct-span policy was simple: open,
  decode SQL under hard grammar control, close as soon as complete, then stop.
- Once `helpers.EndsWithRightDelimiter(generated)` is true, stop immediately.
  Do not enter an after-phase that emits additional helper steps, and do not
  open a second `<< ... >>` span.
- Do not rely on natural LEFT-delimiter nudges for Spider in this mode.
- Do not continue long unconstrained narration after deciding to open the answer
  channel; the run is format-first.
"""


def _append_runtime_mode_reminders(user_prompt: str) -> str:
    user_prompt += _natural_delimiter_user_reminder()
    user_prompt += _scratch_span_preference_reminder()
    user_prompt += _spider_single_sql_span_reminder()
    return user_prompt


def _append_context_block(user_prompt: str, *, heading: str, body: str) -> str:
    body = body.strip()
    if not body:
        return user_prompt
    return user_prompt + f"\n\n{heading}:\n```\n{body}\n```"


def build_initial_prompt(
    task_description: str,
    *,
    strategy_language: str = "python",
    additional_context: str = "",
) -> tuple[str, str]:
    if strategy_language == "dafny":
        user_prompt = DAFNY_INITIAL_GENERATION_PROMPT.format(task_description=task_description)
        user_prompt = _append_runtime_mode_reminders(user_prompt)
        memory_block = build_prompt_memory(task_description, strategy_language=strategy_language)
        if memory_block:
            user_prompt += "\n\n" + memory_block
        user_prompt = _append_context_block(
            user_prompt,
            heading="Additional Run Context",
            body=additional_context,
        )
        return _compose_dafny_system_prompt(), user_prompt

    user_prompt = INITIAL_GENERATION_PROMPT.format(
        task_description=task_description,
        removed_helper_guidance=REMOVED_HELPER_GUIDANCE.strip(),
    )
    user_prompt = _append_runtime_mode_reminders(user_prompt)
    memory_block = build_prompt_memory(task_description, strategy_language=strategy_language)
    if memory_block:
        user_prompt += "\n\n" + memory_block
    user_prompt = _append_context_block(
        user_prompt,
        heading="Additional Run Context",
        body=additional_context,
    )
    return _compose_system_prompt(), user_prompt


def build_verification_error_prompt(
    previous_strategy: str,
    error_message: str,
    *,
    strategy_language: str = "python",
    behavioral_context: str = "",
    structured_feedback: str = "",
    error_history: str = "",
) -> tuple[str, str]:
    behavioral_context_block = ""
    structured_feedback_block = ""
    error_history_block = ""
    if behavioral_context:
        behavioral_context_block = (
            "\nRecent behavioral context from evaluation:\n```\n"
            f"{behavioral_context}\n"
            "```\n\n"
        )
    if structured_feedback:
        structured_feedback_block = (
            "\nStructured verifier analysis:\n```\n"
            f"{structured_feedback}\n"
            "```\n\n"
        )
    if error_history:
        error_history_block = (
            "\nRecent verification history across this run:\n```\n"
            f"{error_history}\n"
            "```\n\n"
        )

    if strategy_language == "dafny":
        user_prompt = DAFNY_VERIFICATION_ERROR_REFINEMENT_PROMPT.format(
            previous_strategy=previous_strategy,
            error_message=error_message,
        )
        user_prompt = _append_context_block(
            user_prompt,
            heading="Recent Behavioral Context From Evaluation",
            body=behavioral_context,
        )
        user_prompt = _append_context_block(
            user_prompt,
            heading="Structured Verifier Analysis",
            body=structured_feedback,
        )
        user_prompt = _append_context_block(
            user_prompt,
            heading="Recent Verification History Across This Run",
            body=error_history,
        )
        return _compose_dafny_system_prompt(), user_prompt

    user_prompt = VERIFICATION_ERROR_REFINEMENT_PROMPT.format(
        previous_strategy=previous_strategy,
        error_message=error_message,
        behavioral_context_block=behavioral_context_block,
        structured_feedback_block=structured_feedback_block,
        error_history_block=error_history_block,
    )
    return _compose_system_prompt(), user_prompt


def build_runtime_error_prompt(
    previous_strategy: str, error_traceback: str, *, strategy_language: str = "python"
) -> tuple[str, str]:
    if strategy_language == "dafny":
        user_prompt = DAFNY_RUNTIME_ERROR_REFINEMENT_PROMPT.format(
            previous_strategy=previous_strategy,
            error_traceback=error_traceback,
        )
        return _compose_dafny_system_prompt(), user_prompt

    user_prompt = RUNTIME_ERROR_REFINEMENT_PROMPT.format(
        previous_strategy=previous_strategy,
        error_traceback=error_traceback,
    )
    return _compose_system_prompt(), user_prompt


def build_compilation_error_prompt(
    previous_strategy: str, error_message: str, *, strategy_language: str = "python"
) -> tuple[str, str]:
    if strategy_language == "dafny":
        user_prompt = DAFNY_VERIFICATION_ERROR_REFINEMENT_PROMPT.format(
            previous_strategy=previous_strategy,
            error_message=error_message,
        )
        return _compose_dafny_system_prompt(), user_prompt

    user_prompt = EVALUATION_FAILURE_REFINEMENT_PROMPT.format(
        previous_strategy=previous_strategy,
        evaluation_feedback=f"(runtime error) {error_message}",
    )
    return _compose_system_prompt(), user_prompt


def build_format_repair_prompt(
    previous_strategy: str, *, strategy_language: str = "python"
) -> tuple[str, str]:
    if strategy_language == "dafny":
        user_prompt = DAFNY_FORMAT_REPAIR_PROMPT.format(previous_strategy=previous_strategy)
        return _compose_dafny_system_prompt(), user_prompt

    user_prompt = FORMAT_REPAIR_PROMPT.format(previous_strategy=previous_strategy)
    return _compose_system_prompt(), user_prompt


def build_evaluation_failure_prompt(
    previous_strategy: str, evaluation_feedback: str, *, strategy_language: str = "python"
) -> tuple[str, str]:
    if strategy_language == "dafny":
        user_prompt = DAFNY_EVALUATION_FAILURE_REFINEMENT_PROMPT.format(
            previous_strategy=previous_strategy,
            evaluation_feedback=evaluation_feedback,
        )
        user_prompt = _append_runtime_mode_reminders(user_prompt)
        return _compose_dafny_system_prompt(), user_prompt

    user_prompt = EVALUATION_FAILURE_REFINEMENT_PROMPT.format(
        previous_strategy=previous_strategy,
        evaluation_feedback=evaluation_feedback,
    )
    user_prompt = _append_runtime_mode_reminders(user_prompt)
    return _compose_system_prompt(), user_prompt


def build_structure_repair_prompt(
    previous_strategy: str, issue: str, *, strategy_language: str = "python"
) -> tuple[str, str]:
    if strategy_language == "dafny":
        user_prompt = DAFNY_STRUCTURE_REPAIR_PROMPT.format(
            previous_strategy=previous_strategy,
            issue=issue,
        )
        user_prompt = _append_runtime_mode_reminders(user_prompt)
        return _compose_dafny_system_prompt(), user_prompt

    user_prompt = STRUCTURE_REPAIR_PROMPT.format(
        previous_strategy=previous_strategy,
        issue=issue,
    )
    user_prompt = _append_runtime_mode_reminders(user_prompt)
    return _compose_system_prompt(), user_prompt
