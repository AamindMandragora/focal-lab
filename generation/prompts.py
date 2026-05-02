"""
Prompt templates for Python-first CSD strategy generation.

The model writes a Python method body for `generation/csd/GeneratedAgentTemplate.py`.
That Python is later transpiled to Dafny for verification.
"""

import os
from functools import lru_cache
from pathlib import Path


PROMPTS_DIR = Path(__file__).resolve().parent
HELPER_REFERENCE_PATH = PROMPTS_DIR / "csd" / "VerifiedAgentSynthesis.md"
CURATED_HELPER_REFERENCE = """\
# Curated Helper Mini-Reference

This mini-reference is distilled from `generation/csd/VerifiedAgentSynthesis.md`.
Use it as the high-signal helper guide when writing the strategy body.

## Core Facts

- `generated` is `list[str]`, not a Python string.
- There is no separate `answer` channel in the current template. The strategy returns one
  `generated` prefix. Delimited spans are parseable islands inside that prefix; every
  span a strategy emits between delimiters must be grammar-valid.
- Do not call deprecated split-channel helper APIs from older templates.
- `helpers.LongestValidSuffix(generated)` is also `list[str]`, not a Python string.
- Never call `generated.startswith(...)`, `generated.endswith(...)`, `generated.strip(...)`,
  or remove delimiters with string slicing. Track phases with booleans/counters instead.
- Never call `.startswith(...)`, `.endswith(...)`, or `.strip(...)` on
  `helpers.LongestValidSuffix(generated)` either.
- `prompt` stays unchanged. Every emitted token goes into `generated`.
- `LeftDelimiter` and `RightDelimiter` are full LM tokens.

## Preferred Step Surface

Prefer these helpers and always assign the tuple result back into `generated, stepsLeft`:

- `helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)`
- `helpers.AppendUnconstrainedAllowLeftDelimiterStep(prompt, generated, stepsLeft)`
- `helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft)`
- `helpers.AppendLeftDelimiter(generated, stepsLeft)`
- `helpers.AppendConstrainedStep(prompt, generated, stepsLeft)`
- `helpers.AppendConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft)`
- `helpers.AppendSoftConstrainedStep(prompt, generated, penalty, stepsLeft)`
- `helpers.AppendTopKConstrainedStep(prompt, generated, k, stepsLeft)`; use a literal `k` of
  `1` unless you have a proof guard for `k <= len(lm.Tokens)`
- `helpers.AppendRightDelimiter(generated, stepsLeft)`

Never use an `Append*` helper as a bare statement.
Never write `next_token, stepsLeft = helpers.AppendConstrainedStep(...)` or any other
`Append*` call into `next_token`; `Append*` returns an updated prefix, not a token.
Use hard `AppendConstrainedStep` for delimited constrained tokens while the grammar suffix is incomplete.
Use `AppendConstrainedOrRightDelimiterStep` when the suffix may be complete and the answer policy
wants the LM to either continue valid grammar tokens or close naturally. Use top-k only as
`AppendTopKConstrainedStep(prompt, generated, 1, stepsLeft)` unless you can prove the bound;
`AppendSoftConstrainedStep` is only a guarded biasing experiment and is not a hard syntax guarantee.
If you use raw `ForcedTokenStep`, always write:
`next_token, new_steps = helpers.ForcedTokenStep(...)`,
then append `next_token`, then set `stepsLeft = new_steps`.
Never manually change `stepsLeft` with `stepsLeft -= 1`, `stepsLeft += 1`, or
`stepsLeft = stepsLeft - ...`; step helpers already consume budget and preserve the proof
invariants.

## Ownership Rules

Grammar-state queries live on `parser`, not on `helpers`.
Budget convenience over the current generated prefix lives on `helpers`.

- Correct: `helpers.CanConstrain(generated)`
- Correct: `helpers.MinStepsToComplete(generated)`
- Correct: `parser.IsCompletePrefix(helpers.LongestValidSuffix(generated))`
- Correct: `parser.ParserDistanceToComplete(helpers.LongestValidSuffix(generated))`
- Correct: `parser.ValidContinuationCount(helpers.LongestValidSuffix(generated))`
- Wrong: `parser.IsCompletePrefix(generated)`
- Wrong: `helpers.ValidContinuationCount(...)`
- Wrong: `parser.IsCompletePrefix(generated)`
- Wrong: `parser.MinStepsToComplete(...)`

## Constrained-Call Rule

Every call to:
- `ConstrainedStep`
- `SoftConstrainedStep`
- `TopKConstrainedStep`
- `AppendConstrainedStep`
- `AppendSoftConstrainedStep`
- `AppendTopKConstrainedStep`

must be inside a branch or loop condition that explicitly mentions
`helpers.CanConstrain(generated)` or
`parser.IsCompletePrefix(helpers.LongestValidSuffix(generated))`.

Do not rely on a phase variable alone.

Guarded branch shape:

```python
elif phase == 2 and helpers.CanConstrain(generated):
    if mode > 0:
        generated, stepsLeft = helpers.AppendTopKConstrainedStep(prompt, generated, 1, stepsLeft)
    else:
        generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
```

Invalid unguarded form:

```python
elif phase == 2:
    generated, stepsLeft = helpers.AppendSoftConstrainedStep(prompt, generated, 0.5, stepsLeft)
```

## Minimal Proof-Friendly Loop Pattern

Place these exact comments immediately above each decoding `while` loop:

```python
# invariant helpers.lm == lm
# invariant helpers.parser == parser
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# invariant |generated| + stepsLeft <= maxSteps
# decreases stepsLeft
```

Every decoding loop must be budget-bounded, for example:
`while stepsLeft > 0 and phase < 3:`
Do not indent invariant or decreases comments inside the loop body; the transpiler only attaches
them to the loop when the comment block is directly above the `while` line.
Declare `phase`, counters, thresholds, and other local state before this proof block. Nothing
except the six invariant/decreases comments may appear between the proof block and the `while`.

## Delimiter Protocol

Any content intended for the parser/evaluator must be emitted inside delimiter spans, and
every delimited span must be grammar-valid. Free-form text outside delimiters is expressive
reasoning and does not need to parse. The final `<< ... >>` span is the default graded answer
unless a dataset-specific evaluator explicitly accumulates earlier spans. Strategies may emit
earlier delimited scratch/logic spans if their state policy deliberately uses them, but this is
an option, not a required template. The structural validator rejects bodies that do not visibly
emit both delimiter tokens.

Use this order:
1. Optional free-form reasoning with `helpers.AppendUnconstrainedStep(...)`
2. `generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)`
3. Append constrained grammar tokens while the answer policy wants more content:
   use `helpers.AppendConstrainedStep(...)` under `helpers.CanConstrain(generated)`;
   if the suffix is complete but extendable, use `helpers.AppendConstrainedStep(...)`
   under `helpers.CanConstrain(generated)`.
4. Only after the grammar suffix is complete and your close policy says the answer is rich enough,
   append `RightDelimiter`

Critical phase rule:
- Do not use `helpers.CanConstrain(generated)` to decide when to leave free-form reasoning before
  emitting `LeftDelimiter`. Free-form text can accidentally have a grammar-shaped suffix; that is
  not the answer channel.
- Enter the answer phase by a phase/budget decision, emit `LeftDelimiter`, and only then use
  `helpers.CanConstrain(generated)` for constrained answer-token calls.
- `parser.IsCompletePrefix(helpers.LongestValidSuffix(generated))` means closing is allowed, not
  required. Do not close just because the prefix first becomes complete. If
  `helpers.CanConstrain(generated)` is true and your answer is still too short, continue
  with `helpers.AppendConstrainedStep(prompt, generated, stepsLeft)` before closing.
- Keep delimiter phases concrete: one phase branch emits only `AppendLeftDelimiter` and advances
  state, later constrained/extend-constrained branches emit answer tokens, and the completion
  branch emits `AppendRightDelimiter` only when completion is true and a separate close policy
  is satisfied. If there are multiple spans, each span must be independently grammar-valid; the
  last span is the default graded answer unless the evaluator says otherwise.
- After `AppendRightDelimiter`, either set a terminal phase if this was the final answer, or
  continue with delimiter-masked unconstrained reasoning and later deliberately open another
  verified span with `AppendLeftDelimiter` and constrained helpers. Do not create raw delimiter
  text by hand; always use delimiter helpers.
- A branch that contains `AppendConstrainedStep`, `AppendSoftConstrainedStep`, or
  `AppendTopKConstrainedStep` is invalid unless that same branch condition textually contains
  `helpers.CanConstrain(generated)`. A boolean variable set from `CanConstrain` is not enough.
- A branch that contains `AppendConstrainedStep` or `ConstrainedStep` must textually
  contain `helpers.CanConstrain(generated)`.

Do not call constrained helpers before executable left-delimiter emission. Do not append both
delimiters after the loop; delimiter calls should live inside explicit budget-bounded phase
branches so `stepsLeft > 0` is visible to the verifier.

These two calls must appear in the executable body unless you intentionally use raw
`ForcedTokenStep` equivalents:

```python
generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)
generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)
```

Mentioning delimiter emission in the rationale or comments is not enough. The literal helper call
must appear as executable Python assigned back to `generated, stepsLeft`.

Do not mention `<<` or `>>` in ordinary reasoning text outside deliberate delimiter emissions.
For GSM arithmetic, preserve decimals exactly; do not turn `8.5` into `8`.
- Do not add a fallback branch that abandons grammar constraints and keeps generating
  unconstrained tokens after entering a delimited constrained segment.
"""

NATURAL_DELIMITER_OVERRIDE = """\
## GSM Natural-Delimiter Mode Override

This run has `CSD_REQUIRE_NATURAL_DELIMITERS=1`. These rules override any older generic
instructions that say to use `AppendLeftDelimiter`, `AppendRightDelimiter`,
`AppendForcedToken`, or `ForcedTokenStep` for delimiter emission.

- Do not call `AppendLeftDelimiter`, `AppendRightDelimiter`, `AppendForcedToken`, or
  `ForcedTokenStep` for GSM delimiters in this mode.
- Use `helpers.AppendUnconstrainedStep(...)` for ordinary reasoning before the first final-answer
  cue; it masks delimiter tokens, which prevents the LM from turning the first local arithmetic
  phrase into the graded span.
- A visible raw-step delimiter decision counts as delimiter emission, but use it only after a
  final-answer cue or meaningful budget pressure:
  `next_token, new_steps = helpers.UnconstrainedAllowLeftDelimiterStep(...)` or
  `helpers.UnconstrainedNudgeLeftDelimiterStep(...)`, then append `next_token`, update
  `stepsLeft`, and switch into the constrained span if
  `next_token == LeftDelimiter or next_token == SpacedLeftDelimiter`.
- Inside a span, use
  `next_token, new_steps = helpers.ConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft)`,
  then append `next_token`, update `stepsLeft`, and close the span if
  `next_token == RightDelimiter or next_token == SpacedRightDelimiter`.
- Track closed verified spans with a real counter such as `closed_spans = 0` or
  `scratch_spans = 0`. Increment it in the right-delimiter branch. Do not use
  `spanTokens`, `answerSteps`, or any token/step counter as the span counter.
- Use that closed-span counter in a loop or branch condition so the strategy can continue after
  a scratch mini-expression and stop after a final answer span. A separate `final_ready` or
  `answer_ready` integer flag is fine.
- Do not use `UnconstrainedAllowLeftDelimiterStep` as the default first-token/free-form reasoning
  step. If it is active from the start, Qwen often emits spans like `In the first 20s,
  <<30 * 1 = 30>>`, which is grammar-valid but not the final answer.
- Once a state variable such as `final_ready`, `answer_ready`, or `late_pressure` says the answer
  span should open, keep using `helpers.UnconstrainedNudgeLeftDelimiterStep(...)` in that branch
  until the LM emits `LeftDelimiter` / `SpacedLeftDelimiter`. Do not switch back to plain
  `UnconstrainedAllowLeftDelimiterStep` after readiness; that often causes 300-token rambles with
  no final delimiter.
- Do not use plain `AppendUnconstrainedStep` / `UnconstrainedStep` as an open-phase fallback after
  several failed nudge attempts. Once the answer-opening phase starts, keep nudging for `<<` while
  budget remains; falling back to ordinary unconstrained reasoning lets the LM finish in prose
  without any graded span.
- Do not wait until only a tiny budget remains, such as `not helpers.HasBudget(stepsLeft, 6)` or
  `stepsLeft <= 4`, before starting the nudge phase. Natural opening needs repeated chances to
  sample `<<`; begin answer-opening pressure with a moderate remaining budget (roughly 16-32
  steps) plus answer intent or scratch-to-final state.
- For GSM, prefer this observed high-performing shape before forcing scratch spans: a substantial
  delimiter-masked reasoning phase, a short wrap-up / answer-cue phase, then persistent
  `AppendUnconstrainedNudgeLeftDelimiterStep` until the first final span opens. Think in
  several-dozen ordinary reasoning/setup steps when `maxSteps` is large enough. If the policy is
  relying mostly on counters rather than a clear final-answer cue, keep the answer-ready threshold
  around the low 40s or later, then spend a few wrap-up/answer-cue steps before nudging. This is
  different from the bad one-short-prefix strategy, and it should inspire new late-opening policies
  rather than copying one fixed counter schedule.
- After a non-final closed span, return to free-form reasoning with delimiter-masked natural
  steps; after the final closed span, set the terminal phase. The policy can still decide that
  the first span is final when the reasoning already supports a complete answer.

Proof-friendly natural control skeleton to adapt, not copy as a fixed template:

```python
phase = 0
closed_spans = 0
reason_signal = 0
final_ready = 0
next_token = eosToken
new_steps = stepsLeft
# invariant helpers.lm == lm
# invariant helpers.parser == parser
# invariant lm.ValidTokensIdsLogits()
# invariant 0 <= stepsLeft <= maxSteps
# invariant |generated| + stepsLeft <= maxSteps
# decreases stepsLeft
while stepsLeft > 0 and phase < 3 and closed_spans < 4:
    if phase == 0:
        if final_ready == 0 and helpers.HasBudget(stepsLeft, 24):
            generated, stepsLeft = helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)
            reason_signal = reason_signal + 1
            if reason_signal > 44 and helpers.HasBudget(stepsLeft, 24):
                final_ready = 1
        else:
            next_token, new_steps = helpers.UnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft)
            generated = generated + [next_token]
            stepsLeft = new_steps
            if next_token == LeftDelimiter or next_token == SpacedLeftDelimiter:
                phase = 1
            else:
                reason_signal = reason_signal + 1
    elif phase == 1 and (helpers.CanConstrain(generated) or parser.IsCompletePrefix(helpers.LongestValidSuffix(generated))):
        next_token, new_steps = helpers.ConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft)
        generated = generated + [next_token]
        stepsLeft = new_steps
        if next_token == RightDelimiter or next_token == SpacedRightDelimiter:
            closed_spans = closed_spans + 1
            phase = 3
        else:
            reason_signal = reason_signal + 1
    else:
        break
```
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

    This keeps the helper surface reference available without anchoring the model to
    concrete worked examples or skeleton strategies.
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
        if skipping:
            continue
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
            + "Use it as the authoritative short reference for helper names, object ownership, and proof-critical usage rules.\n\n"
            + "[BEGIN CURATED_HELPER_REFERENCE]\n"
            + CURATED_HELPER_REFERENCE.strip()
            + "\n[END CURATED_HELPER_REFERENCE]\n"
            + natural_override
        )
    reference = _strip_example_sections(_load_helper_reference_markdown())
    return (
        SYSTEM_PROMPT
        + "\n\n## Additional Authoritative Helper Reference\n\n"
        + "The following markdown is copied from `generation/csd/VerifiedAgentSynthesis.md` with example sections removed.\n"
        + "Use it as the authoritative reference for helper names, signatures, object ownership, and contracts.\n"
        + "If this reference conflicts with your memory, follow the reference.\n\n"
        + "[BEGIN VERIFIED_AGENT_SYNTHESIS_MD]\n"
        + reference
        + "\n[END VERIFIED_AGENT_SYNTHESIS_MD]\n"
        + natural_override
    )


def _natural_delimiter_user_reminder() -> str:
    if not _env_flag("CSD_REQUIRE_NATURAL_DELIMITERS"):
        return ""
    return """\

Natural-delimiter mode reminder:
- For GSM in this run, do NOT use `AppendLeftDelimiter`, `AppendRightDelimiter`,
  `AppendForcedToken`, or `ForcedTokenStep` for delimiters, even if a generic rule above suggests
  them.
- Use `UnconstrainedAllowLeftDelimiterStep` or `UnconstrainedNudgeLeftDelimiterStep` to let the
  LM choose `LeftDelimiter` / `SpacedLeftDelimiter` naturally only after ordinary
  `AppendUnconstrainedStep` reasoning has reached a final-answer cue or budget pressure.
- Once answer-ready, prefer `UnconstrainedNudgeLeftDelimiterStep` over plain
  `UnconstrainedAllowLeftDelimiterStep` until the left delimiter actually appears.
- Use `ConstrainedOrRightDelimiterStep` inside the constrained span, and handle both
  `RightDelimiter` and `SpacedRightDelimiter`.
- Declare a real closed-span counter (`closed_spans` or `scratch_spans`), increment it in the
  right-delimiter branch, and use it in a branch or loop condition. Do not use `spanTokens`.
"""

SYSTEM_PROMPT = """\
You are an expert in formal verification and constrained decoding for language models.
You are generating the BODY of a Python function, not a full file.

The surrounding template is:

  def MyCSDStrategy(lm: LM, parser: Parser, prompt: Prefix, maxSteps: int, eosToken: Token) -> tuple[Prefix, int]:
      helpers = CSDHelpers(lm, parser)
      lm.ValidTokensIdsLogitsAlways()
      generated = []   # all output tokens go here — free-form and constrained alike
      stepsLeft = maxSteps
      [YOUR BODY]
      remainingSteps = stepsLeft
      return generated, remainingSteps

Your output must therefore be ONLY Python statements for [YOUR BODY].
Do not write the function signature, imports, markdown fences, or a full file.
Do not redeclare `helpers`, `generated`, or `stepsLeft`.
Do not assign to `remainingSteps`; the template already does that.
Do not output only comments or only invariants. The body must execute real decoding steps.

## Library API

All tokens, prefixes, and logits are plain Python types (str, list[str], float).
`generated` is a list of token strings, not a Python string. Never call string methods like
`generated.startswith(...)` or `generated.endswith(...)`, and never strip delimiter characters
from it as if `<<` or `>>` were substrings. Delimiters are full tokens; emit them with
`helpers.AppendLeftDelimiter(...)` / `helpers.AppendRightDelimiter(...)` and track phase with
state variables instead of string slicing.

The current helper/template surface is single-prefix, not split-channel. There is no local
`answer` list initialized by the template and no supported split-channel helper APIs. Use
`helpers.AppendUnconstrainedStep(...)` for free-form text and `helpers.AppendConstrainedStep(...)`
or `helpers.AppendConstrainedOrRightDelimiterStep(...)` for grammar-controlled content inside delimiter
spans.

### Raw step functions — consume one step, return (next_token, new_stepsLeft)

- `helpers.UnconstrainedStep(prompt, generated, stepsLeft)` — delimiter-masked free-form reasoning step; use before the strategy is ready to open a verified span.
- `helpers.UnconstrainedAllowLeftDelimiterStep(prompt, generated, stepsLeft)` — masks right delimiters but allows the LM to emit `<<` / ` <<` naturally; if the chosen token is a left delimiter, append it and switch into a constrained phase.
- `helpers.UnconstrainedBiasLeftDelimiterStep(prompt, generated, bias, stepsLeft)` — like `UnconstrainedAllowLeftDelimiterStep`, but positively biases `<<` / ` <<` without forcing it. Use a literal positive bias under real budget pressure when fully natural opening risks missing the format deadline.
- `helpers.UnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft)` — like `UnconstrainedBiasLeftDelimiterStep` with a built-in positive bias. Prefer this over a custom bias variable when format is at risk; it keeps the delimiter LM-chosen without type/verification friction.
- `helpers.ConstrainedStep(prompt, generated, stepsLeft)` — computes `LongestValidSuffix(generated)` to find the current grammar state, masks all invalid tokens, then generates. Use this while the current grammar suffix is incomplete.
- `helpers.ConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft)` — lets the LM choose a grammar-valid continuation token, or `RightDelimiter` / `SpacedRightDelimiter` only when `parser.IsCompletePrefix(helpers.LongestValidSuffix(generated))` is true. Use this when you want `>>` / ` >>` to be LM-chosen without sacrificing the syntax guarantee. Capture `next_token, new_steps`, append `next_token`, update `stepsLeft`, and if `next_token == RightDelimiter or next_token == SpacedRightDelimiter` switch out of the constrained phase.
- `helpers.SoftConstrainedStep(prompt, generated, penalty, stepsLeft)` — like ConstrainedStep but penalizes invalid tokens by `penalty` instead of hard-masking them.
- `helpers.TopKConstrainedStep(prompt, generated, k, stepsLeft)` — grammar-masks first, then applies top-k filtering among valid tokens.
- `helpers.ForcedTokenStep(prompt, generated, token, stepsLeft)` — skips LM generation entirely; emits `token` directly. Use to emit structural tokens like `LeftDelimiter` and `RightDelimiter`. Always capture both return values as `next_token, new_steps = helpers.ForcedTokenStep(...)`, then append `next_token` and update `stepsLeft = new_steps`.

### Preferred append-style helpers — consume one step and return (updated_prefix, remaining_steps)

- `helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)` — preferred wrapper around `UnconstrainedStep`; appends the chosen token for you.
- `helpers.AppendUnconstrainedAllowLeftDelimiterStep(prompt, generated, stepsLeft)` — append wrapper for natural left-delimiter opening.
- `helpers.AppendUnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft)` — append wrapper for biased natural left-delimiter opening.
- `helpers.AppendConstrainedStep(prompt, generated, stepsLeft)` — preferred wrapper around `ConstrainedStep`; appends the grammar-valid token for you.
- `helpers.AppendConstrainedOrRightDelimiterStep(prompt, generated, stepsLeft)` — append wrapper that can emit `>>` / ` >>` after parser completion.
- `helpers.AppendSoftConstrainedStep(prompt, generated, penalty, stepsLeft)` — wrapper around `SoftConstrainedStep`; guarded only, and not a hard syntax guarantee for constrained-span tokens.
- `helpers.AppendTopKConstrainedStep(prompt, generated, k, stepsLeft)` — wrapper around `TopKConstrainedStep`; prefer `k = 1` unless you have a proof guard for `k <= len(lm.Tokens)`.
- `helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft, threshold)` — preferred wrapper around `UnconstrainedStep`.
- `helpers.AppendForcedToken(generated, token, stepsLeft)` — preferred wrapper around `ForcedTokenStep`; appends `token` for you.
- `helpers.AppendLeftDelimiter(generated, stepsLeft)` — append `LeftDelimiter` in one call.
- `helpers.AppendRightDelimiter(generated, stepsLeft)` — append `RightDelimiter` in one call.
- `helpers.Checkpoint(generated)` — capture a reusable prefix checkpoint before risky expansion.
- `helpers.RestoreCheckpoint(checkpoint)` — restore exactly to a saved checkpoint.
- `helpers.RestoreIfDead(generated, checkpoint)` — if the current suffix is dead, revert to checkpoint; otherwise keep current prefix.

When possible, prefer the Append* helpers because they avoid the common proof mistakes around tuple unpacking, forgotten appends, and stale step budgets.

### Which object owns which method

Only a small set of names live on `helpers`. Grammar-state queries like completeness and continuation counts live on `parser`, not on `helpers`.
Budget convenience over the current generated prefix lives on `helpers`.

- Correct: `helpers.CanConstrain(generated)`
- Correct: `helpers.CanConstrain(generated)`
- Correct: `helpers.MinStepsToComplete(generated)`
- Correct: `parser.IsCompletePrefix(helpers.LongestValidSuffix(generated))`
- Correct: `parser.ParserDistanceToComplete(helpers.LongestValidSuffix(generated))`
- Correct: `parser.ValidContinuationCount(helpers.LongestValidSuffix(generated))`
- Wrong: `parser.IsCompletePrefix(generated)`
- Wrong: `helpers.ValidContinuationCount(...)`
- Wrong: `parser.MinStepsToComplete(...)`

If you need parser information about the generated answer segment, first compute the grammar-relevant suffix with `helpers.LongestValidSuffix(generated)` and pass that suffix to `parser`.

### Grammar state queries

- `helpers.LongestValidSuffix(generated)` — returns the longest suffix of `generated` that the parser accepts as a valid prefix. Returns `[]` if no suffix is valid. Use this to check where the constrained segment begins.
- `helpers.CanConstrain(generated)` — shorthand for `not parser.IsCompletePrefix(helpers.LongestValidSuffix(generated))`. Prefer this guard before constrained helper calls.
- `parser.IsValidPrefix(prefix)` — True if `prefix` is a valid partial parse.
- `parser.IsCompletePrefix(prefix)` — True if `prefix` is a complete, finished parse.
- `parser.ValidNextTokens(prefix)` — returns list of valid next tokens from `prefix`.
- `parser.IsDeadPrefix(prefix)` — True if the prefix cannot be extended.
- `parser.ValidContinuationCount(prefix)` — number of valid continuations.
- `parser.ParserDistanceToComplete(prefix)` — lower bound on steps to complete the grammar.
- There is no `parser.MinStepsToComplete(...)`. Use `helpers.MinStepsToComplete(generated)`
  or `parser.ParserDistanceToComplete(helpers.LongestValidSuffix(generated))`.

### Logit shaping (on `lm`, not `helpers`)

- `lm.BiasToken(token, delta)` — add `delta` to one token's logit (clamped to [-1e9, 1e9]).
- `lm.BiasTokens(tokens, delta)` — add `delta` to a list of tokens.
- `lm.ScaleToken(token, factor)` — multiply one token's logit by `factor`.
- `lm.MaskToken(token)` — set one token's logit to -1e9.
- `lm.MaskTokensExcept(tokens)` — mask all tokens except the allowlist.
- `lm.TopKFilter(k)` — keep only the top-k logit tokens; mask the rest.
- `lm.ClampLogits(low, high)` — clamp all logits to [low, high].

### Composite helpers

- `helpers.SoftConstrainToGrammar(prefix, penalty)` — bias invalid tokens by -penalty (no LM call).
- `helpers.IntersectWithGrammar(prefix)` — hard-mask invalid tokens (no LM call).
- `helpers.BiasForCompletion(prefix, bonus)` — bias tokens that would complete the grammar by +bonus.
- `helpers.HasBudget(stepsLeft, needed)` — returns stepsLeft >= needed.
- `helpers.MinStepsToComplete(prefix)` — lower bound on steps to finish from current suffix.
  Pass the current full prefix such as `generated`; this helper extracts the suffix internally.

### Repair and salvage utilities

Repair/salvage helpers are not part of the generated-strategy surface. Do not
use rollback, span extraction, or retry-based repair as a fallback; use explicit
delimiter phases and constrained answer-token steps instead.

### State structures (opt-in)

- `CheckpointStack()` — push/pop/peek saved prefixes for backtracking.
- `RepetitionTracker(ngramSize)` — track n-gram frequencies; call `ApplyRepetitionPenalties(lm)` to bias logits.

### Constants

- `LeftDelimiter = "<<"` and `RightDelimiter = ">>"` are tokens in the LM vocabulary.

## Answer extraction

The evaluator treats delimiter spans as parseable islands in otherwise free-form output. Your
strategy MUST emit `LeftDelimiter` before each constrained span and `RightDelimiter` after it. Prefer the
visible wrapper calls `generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)`
and `generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)`. The structural
validator rejects bodies that only build an `answer` list or only use constrained steps without
delimiter-emission calls. The grammar-constrained content between delimiters is verified; the
final such span is the default graded answer unless a dataset-specific evaluator accumulates
earlier spans too.
For GSM-style arithmetic
tasks, prefer a compact complete expression like `<<16 * 8.5 + 4 * 10.5 + 13>>`;
the evaluator computes its numeric value. Optional earlier spans may define scratch variables
such as `<<x_1 = 48 / 2>>`, and the final span may use them. Do not force this shape when one
final expression is more natural. A standalone numeral is allowed only when the reasoning has
already made it obvious, but expressions are safer and preferred.

## Workflow pattern

A typical strategy body:
1. Generate free-form reasoning with `helpers.AppendUnconstrainedStep(...)` into `generated`.
2. Emit `LeftDelimiter` with `helpers.AppendLeftDelimiter(generated, stepsLeft)`.
3. In a branch whose condition says `helpers.CanConstrain(generated)`, append hard grammar-valid
   answer tokens with `helpers.AppendConstrainedStep(...)` or verifier-friendly
   `helpers.AppendTopKConstrainedStep(prompt, generated, 1, stepsLeft)`.
4. In a branch whose condition says `parser.IsCompletePrefix(helpers.LongestValidSuffix(generated))`,
   emit `RightDelimiter` with `helpers.AppendRightDelimiter(generated, stepsLeft)`. The right
   delimiter is not part of the answer grammar; it is a structural token emitted after completion.

After step 2, `LongestValidSuffix` resets to `[]` (since `<<` is not a grammar token), so the
first `ConstrainedStep` will choose from `ValidNextTokens([])` — the grammar's starting tokens.
Do not jump directly from free-form reasoning to `AppendConstrainedStep` without first emitting
`LeftDelimiter`.
Do not write a transition like "if helpers.CanConstrain(generated): phase = answer" before
`LeftDelimiter`; that makes ordinary reasoning text look like answer grammar state and will be
rejected.
Do not include a fallback branch that switches to unconstrained generation after entering
a delimited constrained segment. If constraints are tight, use soft/top-k constrained helpers or
budget-aware constrained helpers, but keep every delimited segment grammar-controlled.
Do not rely on `AppendSoftConstrainedStep` as the only answer builder: it biases invalid tokens
but does not hard-guarantee the appended token is grammar-valid. Prefer hard constrained or
top-k constrained appends for delimiter-span content.

Safe adaptive constrained branch shape:

```python
elif phase == 2 and helpers.CanConstrain(generated):
    if mode > 0:
        generated, stepsLeft = helpers.AppendTopKConstrainedStep(prompt, generated, 1, stepsLeft)
    else:
        generated, stepsLeft = helpers.AppendConstrainedStep(prompt, generated, stepsLeft)
```

Do not put constrained append calls under a plain `elif phase == 2:` branch; the guard must be
visible on an enclosing `if`, `elif`, or `while` condition.

## Python subset rules

- Use normal Python syntax: `while ...:`, `if ...:`, `and`, `or`, `not`, `==`, `!=`.
- Use Python comments beginning with `#`.
- If you need loop invariants for the Dafny transpiler, put them as comments IMMEDIATELY above the `while`:
    # invariant helpers.lm == lm
    # invariant helpers.parser == parser
    # invariant lm.ValidTokensIdsLogits()
    # invariant 0 <= stepsLeft <= maxSteps
    # invariant |generated| + stepsLeft <= maxSteps
    # decreases stepsLeft
  These use Dafny syntax (`|generated|`, `==>`) even though the executable body uses Python.
- Prefer `len(generated)` in Python; the transpiler lowers it to Dafny length syntax.
- Do not use Python `for` loops. Use `while` loops only.
- Do not use list comprehensions, lambdas, helper functions, or nested function definitions.
- Do not use `break` unless truly necessary.
- If a terminal branch would only update phase/state without consuming `stepsLeft`, use `break`
  so the verifier can prove the `decreases stepsLeft` clause.
- Avoid float state entirely: never write `pressure = 0.5`, `penalty = 0.5`, or compare float
  variables. Use integer counters for control flow and pass literal positive penalties directly
  inside soft-helper calls.
- If you branch between different step choices, predeclare branch outputs before the `if`:
    next_token = eosToken
    new_steps = stepsLeft
    if ...:
        next_token, new_steps = helpers.ConstrainedStep(prompt, generated, stepsLeft)
    else:
        next_token, new_steps = helpers.UnconstrainedStep(prompt, generated, stepsLeft)
- Every emitted token must come from a step call and must consume budget.
- Never manually change `stepsLeft`; do not write `stepsLeft -= 1`, `stepsLeft += 1`, or
  `stepsLeft = stepsLeft - ...`. Helper calls already consume budget, and manual arithmetic
  on `stepsLeft` breaks the standard invariant.
- Never call `helpers.ForcedTokenStep(...)` as a bare statement. Always write:
    next_token, new_steps = helpers.ForcedTokenStep(prompt, generated, LeftDelimiter, stepsLeft)
    generated = generated + [next_token]
    stepsLeft = new_steps
- Never call an `Append*` helper as a bare statement. Always assign its `(updated_prefix, remaining_steps)` result back into `generated` and `stepsLeft`.
- Do NOT call `parser.IsValidPrefix(generated)` or `parser.IsCompletePrefix(generated)` directly.
  Always route grammar queries through `helpers.LongestValidSuffix(generated)` first.

## Required rationale block

Your output MUST begin with:

# CSD_RATIONALE_BEGIN
# <short explanation of the strategy>
# CSD_RATIONALE_END

Then write the Python statements for the body.
"""


INITIAL_GENERATION_PROMPT = """\
Generate a Python strategy body for this use-case:

Use-case description: {task_description}

Requirements:
- Output ONLY the Python body inserted into `MyCSDStrategy`.
- Start with the required rationale block using `#` comments.
- Do not use old split-output helper APIs and no local
  `answer` channel. All emitted tokens go into `generated`.
- Use a `while` loop with these exact invariants as preceding comments:
    # invariant helpers.lm == lm
    # invariant helpers.parser == parser
    # invariant lm.ValidTokensIdsLogits()
    # invariant 0 <= stepsLeft <= maxSteps
    # invariant |generated| + stepsLeft <= maxSteps
    # decreases stepsLeft
- The invariant/decreases comments must be directly above the `while` line, not indented inside
  the loop body after the `while` line.
- Declare all local state variables before the invariant/decreases block. Do not put `phase = ...`,
  counters, or thresholds between the invariant/decreases comments and the `while` line.
- Every decoding `while` loop must be budget-bounded, e.g. `while stepsLeft > 0 and ...:`.
  Do not write open-ended sentinel loops like `while not done:` without a `stepsLeft > 0` guard.
- Do not manually change `stepsLeft`; no `stepsLeft -= 1`, `stepsLeft += 1`, or
  `stepsLeft = stepsLeft - ...`. Helper calls already consume budget.
- Prefer the Append* wrappers unless you genuinely need the raw token return.
- Before any `ConstrainedStep`, `SoftConstrainedStep`, `TopKConstrainedStep`,
  `AppendConstrainedStep`, `AppendSoftConstrainedStep`, or `AppendTopKConstrainedStep` call,
  ensure the current grammar suffix is incomplete, preferably with `helpers.CanConstrain(generated)`.
- Before any `ConstrainedStep` or `AppendConstrainedStep` call, ensure there is at
  least one valid continuation with `helpers.CanConstrain(generated)`.
- Put constrained helper calls inside a branch whose condition explicitly mentions
  `helpers.CanConstrain(generated)`. Do not rely on some earlier phase variable alone.
- For adaptive soft/top-k/hard policies, put the guard on the enclosing branch, then choose the
  variant inside it; e.g. `elif phase == 2 and helpers.CanConstrain(generated):`.
- Use positional helper arguments only. Write
  `helpers.AppendTopKConstrainedStep(prompt, generated, 1, stepsLeft)`, not keyword arguments.
- Avoid top-k values larger than `1`; the verifier usually cannot prove `k <= len(lm.Tokens)`.
- Avoid float state in control flow. Never assign `pressure = 0.5`, `penalty = 0.5`, or any
  other float local; use integer counters and pass literal penalties directly inside soft-helper
  calls such as `helpers.AppendSoftConstrainedStep(prompt, generated, 0.5, stepsLeft)`.
- For `helpers.UnconstrainedBiasLeftDelimiterStep`, pass a literal positive float such as `5.0`
  directly in the call. Do not write `biasStrength = 3` or pass an integer/local variable.
- Do not write parser wrappers on `helpers` that do not exist; use `parser.*` or the supported helper wrappers.
  Those are parser calls, not helper calls.
- Do not write `parser.MinStepsToComplete(...)`. Use `helpers.MinStepsToComplete(generated)` or
  `parser.ParserDistanceToComplete(helpers.LongestValidSuffix(generated))`.
- The strategy MUST emit `LeftDelimiter` (prefer `helpers.AppendLeftDelimiter(...)`), followed by
  grammar-constrained tokens (prefer `helpers.AppendConstrainedStep(...)`), followed by
  `RightDelimiter` (prefer `helpers.AppendRightDelimiter(...)`).
- Do not use `helpers.CanConstrain(generated)` to decide when to emit `LeftDelimiter`; emit
  `LeftDelimiter` when your state/budget policy decides a constrained span should begin.
  Free-form text can accidentally have a grammar-shaped suffix, and that suffix is not the
  delimited segment.
- Use explicit delimiter phases: a left-delimiter phase branch emits
  `AppendLeftDelimiter` and advances state; answer phase branches guarded by
  `helpers.CanConstrain(generated)` emit answer
  tokens; a completion branch guarded by
  `parser.IsCompletePrefix(helpers.LongestValidSuffix(generated))` plus an explicit close policy
  emits `AppendRightDelimiter` and exits.
- Completion is a permission to close, not an instruction to close immediately. In natural
  delimiter mode, use `helpers.AppendConstrainedOrRightDelimiterStep(...)` in the span so the LM
  can either continue a valid grammar token or emit `>>` once completion is reached.
- In the constrained answer phase, check completion before open-ended extension. A branch like
  `elif phase == 2 and helpers.CanConstrain(generated): ...` before the complete-prefix branch
  can keep extending a complete expression forever because many complete expressions are also
  valid prefixes. Either put the `parser.IsCompletePrefix(helpers.LongestValidSuffix(generated))`
  close/extend branch first, or add
  `and not parser.IsCompletePrefix(helpers.LongestValidSuffix(generated))` to the open-ended
  constrained branch.
- Any branch containing `AppendConstrainedStep`, `AppendSoftConstrainedStep`, or
  `AppendTopKConstrainedStep` must have `helpers.CanConstrain(generated)` in that branch's own
  condition line. Do not store it in `can_constrain` and do not rely on `phase` alone.
- Any branch containing `AppendConstrainedStep` must have
  `helpers.CanConstrain(generated)` in that branch's own condition line.
- Include both visible delimiter calls in executable branches unless using raw
  `helpers.ForcedTokenStep(prompt, generated, LeftDelimiter, stepsLeft)` and
  `helpers.ForcedTokenStep(prompt, generated, RightDelimiter, stepsLeft)` equivalents.
- Saying "emit the left delimiter" in the rationale or comments is not enough; the executable
  body must contain `generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)`
  or an equivalent raw forced-token sequence.
- Emit the left delimiter before any constrained answer-token helper call. Do not put both
  delimiter calls after the loop.
- You must ensure `parser.IsCompletePrefix(helpers.LongestValidSuffix(generated))` before emitting
  `RightDelimiter`. In explicit-delimiter mode the strategy emits it with `AppendRightDelimiter`;
  in natural-delimiter mode use `ConstrainedOrRightDelimiterStep` /
  `AppendConstrainedOrRightDelimiterStep` so `>>` is allowed only after completion.
  The final grammar-constrained content between delimiters is the graded answer.
- After emitting `RightDelimiter`, you may either stop if this was the final answer span, or
  continue with delimiter-masked unconstrained reasoning and later open another verified span
  with `AppendLeftDelimiter` plus constrained helpers. This is the preferred GSM pattern when
  intermediate arithmetic facts are useful.
- Do not emit `RightDelimiter` merely because the suffix first became complete. Use adaptive
  state such as grammar distance, continuation count, budget pressure, or a `close_ready` flag
  derived from semantic progress so the answer can continue through
  `AppendConstrainedOrRightDelimiterStep` while valid continuations exist. Avoid tiny fixed
  `min_reason_steps`, `min_answer_steps`, or similar phase quotas that open or close after only a
  handful of tokens. For GSM, a durable delayed free-form phase can be useful before the final
  answer span; the bad pattern is a short quota that captures the first local arithmetic fragment.
- Put `RightDelimiter` emission inside a branch whose condition explicitly mentions
  `parser.IsCompletePrefix(helpers.LongestValidSuffix(generated))`.
- If you use integer phase values for a terminal or close-ready state, keep them inside the
  loop guard. For example, do not set `phase = 22` inside `while phase < 3`; instead use
  `phase = 3`, or set a separate `close_ready` flag and close in a complete-prefix branch.
- If you use raw `ForcedTokenStep`, always capture it as `next_token, new_steps = ...`, append
  `next_token`, and then update `stepsLeft = new_steps`. Do not append `LeftDelimiter` or
  `RightDelimiter` literals directly.
- The strategy may generate free-form reasoning before entering the constrained segment; use
  `helpers.AppendUnconstrainedStep(...)` or `helpers.UnconstrainedStep(...)` for that.
- If the task is GSM-style arithmetic, prefer policies that can interleave free-form reasoning
  with grammar-verified delimited arithmetic spans. A useful strategy may reason freely, open a
  delimited span for a complete subexpression or scratch assignment, close it, continue reasoning,
  and finally emit a complete delimited answer expression. However, the strongest simple GSM
  baseline is often a single late final span: keep delimiters masked during substantial
  free-form reasoning, add a short wrap-up / answer-cue phase, then repeatedly nudge for a natural
  left delimiter and stop after the first closed final span. Large delayed free-form counters are
  acceptable when paired with budget checks; avoid tiny quotas that open after the first local
  calculation or fixed constrained-token quotas that close before the expression is semantically
  complete.
  Because `AppendUnconstrainedStep` intentionally masks delimiter tokens and hides the emitted
  token from the strategy, use it for ordinary pre-answer reasoning when you do not yet want a
  delimiter. Do not let `UnconstrainedStep` run from the first token: that often
  captures the first local arithmetic fragment as the final answer. Once there is an explicit
  final-answer cue, scratch-to-final transition, or real budget pressure, use a raw observed step:
  `helpers.UnconstrainedAllowLeftDelimiterStep(...)` or
  `helpers.UnconstrainedNudgeLeftDelimiterStep(...)`, then append `next_token`; if it is
  `LeftDelimiter` or `SpacedLeftDelimiter`, switch into the constrained phase without forcing a
  left delimiter. Inside a delimited span, prefer `helpers.ConstrainedOrRightDelimiterStep(...)`
  when the strategy wants the LM to decide naturally whether to continue the expression or close
  with `>>`; that helper only permits `RightDelimiter` / `SpacedRightDelimiter` after parser completion.
  When checking whether the span closed, handle both `next_token == RightDelimiter` and
  `next_token == SpacedRightDelimiter` (or the literal string `" >>"`). Otherwise, let
  natural reasoning milestones influence later phase changes:
  punctuation, newline-like tokens, words such as "therefore", "total", "answer", or budget
  pressure. Do not open `<<` merely because `reasoning_seen` became nonzero or because
  `helpers.HasBudget(...)` is true; that creates outputs like `To<<...>>` / `The<<...>>` and
  usually constrains the first local arithmetic fragment instead of the answer.
  In natural-delimiter mode, do not call `AppendLeftDelimiter`, `AppendRightDelimiter`,
  `AppendForcedToken`, or `ForcedTokenStep` for delimiters. Also handle both natural left delimiter
  tokenizations: `LeftDelimiter` and `SpacedLeftDelimiter` (or the literal string `" <<"`), and both
  natural right delimiter tokenizations: `RightDelimiter` and `SpacedRightDelimiter` (or `" >>"`).
  Natural-delimiter GSM strategies must include executable calls to BOTH
  `helpers.UnconstrainedAllowLeftDelimiterStep(...)` or
  `helpers.UnconstrainedNudgeLeftDelimiterStep(...)` before the span, AND
  `helpers.ConstrainedOrRightDelimiterStep(...)` inside the span. Mentioning them in rationale is
  not enough.
  If evaluation feedback shows missing delimiters or outputs that run to the max token cap before
  opening `<<`, keep the decision natural but switch from
  `UnconstrainedAllowLeftDelimiterStep` to
  `UnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft)` under budget pressure. This
  biases the delimiter without post-hoc wrapping or
  forced delimiter emission. Budget pressure must leave room for the whole verified span; do not
  wait until the last 3-5 tokens before nudging, because the strategy still needs to emit `<<`, a
  complete expression, and `>>`.
  Avoid trivial "one short prefix, one constrained answer span" policies; they verify but usually
  only capture the first local subproblem. A nontrivial one-span policy is still acceptable when it
  has a durable delayed reasoning / wrap-up / answer-opening schedule before the span, because then
  the single span is likely to be the final answer rather than a scratch fragment. Prefer state
  machines that can revisit a reasoning phase after closing an intermediate span when scratch spans
  are actually used, and keep a small counter of verified spans emitted.
  Intermediate spans should be semantically reusable: prefer scratch assignments such as
  `x_1 = 16 * 8.5` or `total_1 = 4 * 12 + 5 * 85` over anonymous fragments like `16 * 8.5 + 4`.
  The final span should compose the scratch values and any remaining constants.
  Strong default pattern for multi-step GSM:
  reason about a useful quantity, let the LM naturally open `<<`, emit a complete assignment such
  as `x_1 = 16 * 8.5`, naturally close `>>`, return to free-form reasoning, optionally bind another
  useful quantity such as `x_2 = 4 * 10.5`, and end with a final answer-bearing span such as
  `x_1 + x_2 + 13`. This is only a preferred pattern, not a rigid template: use it when it helps,
  but every parsed mini-expression must still be complete and delimiter-contained.
  Do not let the first scratch assignment be the final output unless it truly is the answer; if an
  intermediate span binds `x_1`, the strategy should continue reasoning and later emit a final span
  that references `x_1`.
  Prefer a short complete arithmetic expression/equation in the final span. The GSM CSD grammar
  intentionally rejects lone numerals such as `1` or `8` and first-operation fragments such as
  `16 * 8`; include at least one top-level plus/minus clause, e.g. `8 + 0`, when the direct
  answer is already known. The evaluator computes the expression. Optional earlier delimited
  spans may introduce scratch variables such as `x_1 = 48 / 2`, but use that only when it emerges
  naturally from the strategy's state policy; do not force every answer into a fixed scratchpad.
- For GSM-style arithmetic, preserve the numeric values from the problem exactly. Do not round or
  truncate decimals like `8.5` into `8`.
- Novelty requirements:
  - Do NOT produce a trivial two-phase "all unconstrained then all constrained" loop with no
    adaptive control.
  - Do NOT use an unconstrained fallback branch inside or after a delimited constrained segment.
  - Use multiple state variables to drive adaptive decisions across phases.
  - At least two interacting signals must affect what step type is chosen each iteration.
  - Favor strategies that evolve their constraint strength over time (e.g., soft → hard, or
    grammar-distance-aware budgeting).
- Maintain at least two extra local state variables beyond `generated`, `stepsLeft`, `next_token`,
  and `new_steps`.
- Do NOT call parser methods on `generated` directly; always use
  `helpers.LongestValidSuffix(generated)` to route grammar queries.
"""


VERIFICATION_ERROR_REFINEMENT_PROMPT = """\
Your previous Python strategy body failed verification after being transpiled to Dafny.

Previous attempt:
```python
{previous_strategy}
```

Verification error:
```
{error_message}
```

Fix the Python body while preserving the overall strategy when possible.

Rules:
- Output ONLY a corrected Python body, not a full file.
- Start with the required rationale block using `#` comments.
- Keep the strategy within the supported Python subset from the system prompt.
- Use `# invariant ...` and `# decreases ...` comments immediately above `while` loops.
- Keep the standard proof-carrying loop lines unless there is a strong reason to strengthen them:
    # invariant helpers.lm == lm
    # invariant helpers.parser == parser
    # invariant lm.ValidTokensIdsLogits()
    # invariant 0 <= stepsLeft <= maxSteps
    # invariant |generated| + stepsLeft <= maxSteps
    # decreases stepsLeft
- Every decoding `while` loop must mention the decreasing budget in its condition, e.g.
  `while stepsLeft > 0 and ...:`.
- Do not call `AppendConstrainedStep` once `parser.IsCompletePrefix(helpers.LongestValidSuffix(generated))`
  is true. If the answer is complete but still too short and has valid continuations, use
  `helpers.AppendConstrainedStep(...)` under `helpers.CanConstrain(generated)`.
- Completion means right-delimiter emission is allowed, not mandatory. Do not immediately close
  on the first complete prefix unless a separate close policy is satisfied.
- In a `# decreases stepsLeft` loop, every top-level branch must either consume a helper step or
  `break`; do not use branches that only assign phase/state and loop again.
- Do not redeclare `helpers`, `generated`, or `stepsLeft`.
- Do not assign to `remainingSteps`.
- Do not manually change `stepsLeft`; remove `stepsLeft -= 1`, `stepsLeft += 1`, and
  `stepsLeft = stepsLeft - ...`. Helper calls already consume budget and preserve the
  budget invariants.
- Remove keyword arguments from helper calls; all helper calls should be positional.
- If `AppendTopKConstrainedStep` fails because of `1 <= k <= |lm.Tokens|`, replace it with
  `AppendConstrainedStep` or use literal `k = 1`.
- If verification complains that `decreases stepsLeft` may not decrease, replace phase-only
  terminal branches with `break` or make that branch consume a helper step.
- Ensure `AppendLeftDelimiter` happens before constrained answer-token helpers, and emit
  `AppendRightDelimiter` only under
  `parser.IsCompletePrefix(helpers.LongestValidSuffix(generated))`.
- After `AppendRightDelimiter`, either terminate or continue delimiter-masked/free-form reasoning
  before explicitly opening another verified span. Do not let raw `>>` appear outside a completed
  constrained span.

Common fixes:
- If you used Dafny syntax like `:=`, replace it with Python `=`.
- If you used `&&`, `||`, or `!`, replace them with `and`, `or`, and `not`.
- If you used `//` comments, replace them with `#` comments.
- If the verifier complained that branch-local step outputs were undefined, predeclare `next_token`
  and `new_steps` before the `if`.
- Do NOT call parser methods on `generated` directly; use `helpers.LongestValidSuffix(generated)`.
- The repaired body must still emit `LeftDelimiter`, constrained tokens, and `RightDelimiter`.
"""


RUNTIME_ERROR_REFINEMENT_PROMPT = """\
Your previous Python strategy body failed at runtime.

Previous attempt:
```python
{previous_strategy}
```

Runtime traceback:
```
{error_traceback}
```

Produce a corrected Python body only.
Keep the rationale block at the top.
Stay within the supported Python subset.
Prefer minimal changes that preserve the strategy idea.
The fixed body must emit `LeftDelimiter`, grammar-constrained tokens, and `RightDelimiter`.
"""


EVALUATION_FAILURE_REFINEMENT_PROMPT = """\
Your previous Python strategy body verified and ran, but it performed poorly on evaluation.

Previous attempt:
```python
{previous_strategy}
```

Evaluation feedback:
```
{evaluation_feedback}
```

Produce a revised Python body only.
Keep the rationale block at the top.
Try a meaningfully different constrained-decoding strategy if the current one is not working.
If the task is GSM-style arithmetic, the final `<< >>` segment may stay as a short expression or
equation instead of simplifying to a standalone numeral. It may also use optional scratch values
defined in earlier grammar-valid delimited spans, but do not force a fixed scratchpad template.
If the raw GSM outputs look like `To<<...>>`, `The<<...>>`, or otherwise open a delimiter after
one or two generic words, the strategy opened the constrained span too early. Revise it to observe
raw unconstrained tokens. Prefer `helpers.UnconstrainedAllowLeftDelimiterStep` or
`helpers.UnconstrainedNudgeLeftDelimiterStep` so the LM can
naturally emit the left delimiter; if it does, append it and enter constrained decoding. Otherwise
wait for natural reasoning milestones such as punctuation/newline/therefore/total/answer or real
budget pressure before opening a verified span. Do not replace the failure with
`reasoning_seen == 1` plus `helpers.HasBudget(...)`; that is the same early-opening bug in another
form. Similarly, prefer `helpers.ConstrainedOrRightDelimiterStep` inside the span if the failure is
premature or awkward forced `>>`; it lets the LM choose `>>` naturally, but only once the parser
prefix is complete. This is different from adding a fixed minimum token quota.
If the raw GSM outputs open around the first local calculation, e.g. `In the next 20s,
<<2 * 30 = 60>>` when the gold answer needs several periods, the strategy is still opening too
early. Do not use low token thresholds such as `reason_tokens >= 10` or a simple
`closed_spans >= 2` terminal rule as a proxy for finality. Prefer one of two behaviors:
1. keep reasoning unconstrained for a durable late-answer phase, then let the next delimiter be
natural after the model has had room to write an explicit final-answer cue such as `final`,
`answer`, `total`, `altogether`, `therefore`, or `so the answer`. A useful GSM repair can delay
for several dozen ordinary free-form/setup steps, add a short wrap-up / answer-cue phase, then
repeatedly nudge for the first final span; or
2. if an early span is allowed, make it a named scratch assignment (`x_1 = ...`) and require a
later final span that composes scratch values and remaining constants.
The last span should be the composed answer expression, not the second scratch/local fragment.
If format fails because no `<<` appears before max steps, use
`helpers.UnconstrainedNudgeLeftDelimiterStep(prompt, generated, stepsLeft)` under budget
pressure rather than forcing `AppendLeftDelimiter`. Interpret budget pressure early enough to leave
room for the constrained expression and closing delimiter, not only at the final few tokens.
Do not reopen a new constrained window solely because `stepsLeft` is small; combine budget pressure
with explicit final-answer intent/state (for example `final_ready` plus span history).
Do not use parser-distance or valid-continuation predicates as finality signals after only a
short setup; those predicates know grammar shape, not whether the math answer is semantically
final. Also do not set `phase = "open"` / `"span"` and then immediately `break` before a helper
step; that exits the loop with free-form text instead of producing the final `<< ... >>` span.
The strategy must still emit `LeftDelimiter`, grammar-constrained tokens, and `RightDelimiter`.
"""


FORMAT_REPAIR_PROMPT = """Your output must be a Python method body. It is missing the required rationale block markers.

Rewrite the following body so that it starts with:
# CSD_RATIONALE_BEGIN
# ...
# CSD_RATIONALE_END

Then keep the Python body statements.

Previous output:
```python
{previous_strategy}
```

Output ONLY the corrected Python body, no markdown fences.
"""


STRUCTURE_REPAIR_PROMPT = """\
The last Python strategy body was structurally invalid for this project.
The invalid body is intentionally not shown; do not copy or minimally edit the previous shape.
Write a fresh strategy body that satisfies the issue and rules below.

Issue:
{issue}

Write a valid Python method body.

If the issue says the body must emit both delimiters, repair by adding explicit executable
phase branches inside the budget-bounded decoding loop instead of inventing an `answer` channel.
Do not paste delimiter calls after the loop.

Rules:
- Output ONLY the body, not a full file.
- Keep the rationale block at the top.
- Include executable decoding logic, not just comments.
- Do not use old split-output helper APIs or a separate `answer`
  channel; this template returns one `generated` prefix.
- Every decoding `while` loop must be budget-bounded, e.g. `while stepsLeft > 0 and ...:`.
- Do not manually change `stepsLeft`; remove `stepsLeft -= 1`, `stepsLeft += 1`, and
  `stepsLeft = stepsLeft - ...`. Helper calls already consume budget.
- Prefer `helpers.CanConstrain(generated)` over spelling out the full suffix-complete check.
- Include the standard proof-carrying loop lines:
  `# invariant helpers.lm == lm`,
  `# invariant helpers.parser == parser`,
  `# invariant lm.ValidTokensIdsLogits()`,
  `# invariant 0 <= stepsLeft <= maxSteps`,
  `# invariant |generated| + stepsLeft <= maxSteps`,
  and `# decreases stepsLeft`.
- Put those proof-carrying lines directly above the `while` line. Do not indent them inside the
  loop body.
- Put every local state declaration above the proof-carrying lines. Do not place `phase = ...`,
  counters, thresholds, or any executable statement between `# decreases stepsLeft` and `while`.
- Prefer the Append* helpers:
  `helpers.AppendLeftDelimiter(generated, stepsLeft)`,
  `helpers.AppendConstrainedStep(prompt, generated, stepsLeft)`,
  `helpers.AppendConstrainedStep(prompt, generated, stepsLeft)`,
  `helpers.AppendRightDelimiter(generated, stepsLeft)`.
- If the issue says `Missing: LeftDelimiter`, put the actual executable call
  `generated, stepsLeft = helpers.AppendLeftDelimiter(generated, stepsLeft)` in a phase branch.
- If the issue says `Missing: RightDelimiter`, put the actual executable call
  `generated, stepsLeft = helpers.AppendRightDelimiter(generated, stepsLeft)` in a completion
  guarded phase branch.
- One phase branch should emit the left delimiter, a later guarded phase branch should emit
  constrained or extend-constrained answer tokens, and a final completion-guarded phase branch
  should emit the right delimiter. All three branches must live inside the decoding loop.
- The left-delimiter branch should be controlled by phase/budget state, not by
  `helpers.CanConstrain(generated)`.
- If you have `phase == 1`, use it for executable left-delimiter emission before any constrained
  helper call; do not make `phase == 1` the constrained-token phase unless an earlier phase
  branch already emitted `AppendLeftDelimiter`.
- Emit the left delimiter before any constrained answer-token helper call, and emit the right
  delimiter only inside a branch guarded by
  `parser.IsCompletePrefix(helpers.LongestValidSuffix(generated))`.
- Completion is only permission to close. Do not switch to the right-delimiter phase solely
  because `parser.IsCompletePrefix(...)` became true; require an adaptive close policy based on
  grammar distance, continuation count, budget pressure, or semantic progress, or continue with
  `helpers.AppendConstrainedStep(...)` while `helpers.CanConstrain(generated)`.
- In a `# decreases stepsLeft` loop, every top-level `if`/`elif`/`else` branch must either call
  a helper step that consumes budget or `break`. Do not write branches that only assign
  `phase = ...`, `close_ready = ...`, or other state and then loop again.
- Do not put delimiter calls after the loop where `stepsLeft` may already be zero.
- Use positional helper arguments only; remove `k=` or `stepsLeft=` from helper calls.
- Avoid `AppendTopKConstrainedStep` with `k > 1`; prefer `AppendConstrainedStep(...)` or
  `AppendTopKConstrainedStep(prompt, generated, 1, stepsLeft)`.
- Avoid float control state. Never assign `pressure = 0.5`, `penalty = 0.5`, or any other float
  local; use integer counters and pass literal soft penalties directly in soft-helper calls.
- Keep parser queries on `parser`, not on `helpers`. For example, write
  `parser.ValidContinuationCount(helpers.LongestValidSuffix(generated))`, not
  `helpers.ValidContinuationCount(...)`.
- Do not call `parser.MinStepsToComplete(...)`; use `helpers.MinStepsToComplete(generated)` or
  `parser.ParserDistanceToComplete(helpers.LongestValidSuffix(generated))`.
- Only call constrained helpers while `helpers.CanConstrain(generated)`.
- Put each constrained-helper call in a branch whose condition explicitly mentions
  `helpers.CanConstrain(generated)`.
- Do not repair an unguarded constrained call by setting a local boolean such as
  `can_constrain = helpers.CanConstrain(generated)`; the guard must be visible in the `if`/`elif`
  condition that encloses the helper call.
- If the issue mentions an unguarded constrained helper, use this shape:
  `elif phase == 2 and helpers.CanConstrain(generated):`
  then nest adaptive choices such as soft/top-k/hard inside that branch.
- Never put `AppendSoftConstrainedStep`, `AppendConstrainedStep`, or
  `AppendTopKConstrainedStep` under a plain `elif phase == 2:` branch.
- If you use raw forced-token calls, always capture them with `next_token, new_steps = ...`.
- Assign every Append* result back into `generated` and `stepsLeft`.
- Do not use repair/salvage helpers as fallback control flow; keep the answer path explicit.
- If you use raw step calls, append every emitted token with `generated = generated + [next_token]`
  rather than appending delimiter literals directly, and update the budget with `stepsLeft = new_steps`.
- Do NOT call parser methods on `generated` directly.
- Maintain at least two extra local state variables that materially affect control flow.
"""


# Narrow public strategy surface used after pruning experimental helpers from
# CSDHelpers. The older prompt constants above are kept only as historical text
# in this file; these reassignments are the runtime source of truth.
CURATED_HELPER_REFERENCE = """\
# Curated Helper Mini-Reference

Use these helpers exactly as named. Do not invent helper names.

## Core State

- `generated` is `list[str]`, not a string.
- All emitted tokens go into `generated`.
- Emit answer-bearing tokens inside `<< ... >>`.

## Step Helpers

- `helpers.AppendUnconstrainedStep(prompt, generated, stepsLeft)`
- `helpers.AppendConstrainedStep(prompt, generated, stepsLeft)`
- `helpers.AppendSoftConstrainedStep(prompt, generated, penalty, stepsLeft)`
- `helpers.AppendTopKConstrainedStep(prompt, generated, k, stepsLeft)`
- `helpers.ForcedTokenStep(prompt, generated, token, stepsLeft)`
- `helpers.AppendForcedToken(generated, token, stepsLeft)`
- `helpers.AppendLeftDelimiter(generated, stepsLeft)`
- `helpers.AppendRightDelimiter(generated, stepsLeft)`

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
- `helpers.IsLeftDelimiterToken(token)`
- `helpers.IsRightDelimiterToken(token)`

## Logit-Shaping Composites

- `helpers.SoftConstrainToGrammar(generated, penalty)`
- `helpers.IntersectWithGrammar(generated)`
- `helpers.BiasForCompletion(generated, bonus)`
- `helpers.MaskAllDelimiters(generated)`
- `helpers.MaskLeftDelimiters(generated)`
- `helpers.MaskRightDelimiters(generated)`
- `helpers.BiasLeftDelimiters(bias)`
- `helpers.BiasRightDelimiters(bias)`

## Checkpoint And Budget

- `helpers.Checkpoint(generated)`
- `helpers.RestoreCheckpoint(checkpoint)`
- `helpers.RestoreIfDead(generated, checkpoint)`
- `helpers.HasBudget(stepsLeft, needed)`
"""

NATURAL_DELIMITER_OVERRIDE = """\
## Delimiter Policy Override

If this run enables a delimiter override, keep delimiter control explicit:

- Open constrained spans with `AppendLeftDelimiter(...)`.
- Emit constrained tokens with `AppendConstrainedStep(...)` (or guarded soft/top-k variants).
- Emit `AppendRightDelimiter(...)` only after completion checks.
- Do not emit raw delimiter string literals directly.
"""

SYSTEM_PROMPT = """\
You are generating the BODY of a Python function for a verified constrained
decoding strategy. Output only Python statements for the body: no imports, no
function signature, no markdown fences.

The surrounding template already defines:

  helpers = CSDHelpers(lm, parser)
  generated = []
  stepsLeft = maxSteps

Do not redeclare those names and do not return manually.

Your body must contain a `# CSD_RATIONALE_BEGIN` / `# CSD_RATIONALE_END` comment
block, then executable decoding code.

Every decoding `while` loop must have this exact comment block immediately above
it:

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

Use only the curated helper surface below. Prefer helper wrappers such as
`helpers.IsComplete(generated)` and `helpers.ValidContinuationCount(generated)`
over direct parser calls.
"""

INITIAL_GENERATION_PROMPT = """\
Generate a Python strategy body for this use-case:

Use-case description: {task_description}

Requirements:
- Output ONLY the Python body inserted into `MyCSDStrategy`.
- Start with the required `# CSD_RATIONALE_BEGIN` / `# CSD_RATIONALE_END`
  rationale block.
- Use a budget-bounded `while` loop with the exact invariant block described in
  the system prompt.
- Use only the curated helper surface. Prefer append-style helpers.
- Do not call removed helpers: no soft constrained steps, no top-k constrained
  steps, no budget-aware switching, no extend-constrained helpers, no rollback
  or salvage helpers, no repetition structures, and no direct LM logit shaping.
- Avoid checkpoint rollback by default. `RestoreCheckpoint` and `RestoreIfDead`
  require extra checkpoint-length invariants and are verifier-hostile for GSM
  natural-delimiter strategies. Prefer one step-consuming helper call per loop
  iteration and let the next iteration inspect `IsDead`, `EndsWithLeftDelimiter`,
  and `EndsWithRightDelimiter`.
- Prefer helper parser wrappers: `helpers.IsComplete(generated)`,
  `helpers.ValidContinuationCount(generated)`,
  `helpers.ParserDistanceToComplete(generated)`, and
  `helpers.MinStepsToComplete(generated)`.
- Do not manually change `stepsLeft`; no `stepsLeft -= 1`, `stepsLeft += 1`,
  or `stepsLeft = stepsLeft - ...`. Helper calls already consume budget.
- For natural delimiter mode, ordinary reasoning should use
  `helpers.AppendUnconstrainedStep(...)`; answer-opening pressure should use
  `helpers.AppendUnconstrainedNudgeLeftDelimiterStep(...)`; constrained spans
  should use `helpers.AppendConstrainedOrRightDelimiterStep(...)`. Detect
  span boundaries with `helpers.EndsWithLeftDelimiter(generated)` and
  `helpers.EndsWithRightDelimiter(generated)`. In natural mode, avoid plain
  `AppendConstrainedStep`; it extends expressions but cannot emit the closing
  delimiter.
- Start natural answer-opening early enough to have several nudge attempts; do
  not wait for thresholds like `stepsLeft <= 4` or `not helpers.HasBudget(stepsLeft, 6)`.
- Natural mode must keep an explicit span-state variable. Set it when
  `EndsWithLeftDelimiter` becomes true, keep using
  `AppendConstrainedOrRightDelimiterStep` while it is active, and clear it only
  after `EndsWithRightDelimiter`.
- Inside the span, prefer a positive guard:
  `helpers.IsComplete(generated) or helpers.CanConstrain(generated)`, then call
  `AppendConstrainedOrRightDelimiterStep`. Avoid separate early
  `not helpers.CanConstrain` branches.
- Maintain at least two meaningful local state variables that affect control
  flow. Novel control policies are encouraged, but all parser-handled content
  must be emitted inside grammar-valid delimiter spans.
"""

VERIFICATION_ERROR_REFINEMENT_PROMPT = """\
The previous strategy failed verification or transpilation.

Previous strategy:
{previous_strategy}

Error:
{error_message}

Rewrite the strategy body using only the curated helper surface. Preserve the
natural-delimiter style when applicable. Prefer helper wrappers such as
`helpers.IsComplete(generated)` over direct parser calls. Do not use removed
helpers such as soft/top-k/budget-aware/extend-constrained/rollback helpers.
Return only the corrected Python body with the required rationale block.
"""

RUNTIME_ERROR_REFINEMENT_PROMPT = """\
The previous strategy failed at runtime.

Previous strategy:
{previous_strategy}

Traceback:
{error_traceback}

Rewrite the strategy body using only the curated helper surface. Assign every
Append* helper result back into `generated, stepsLeft`; raw token-returning
steps must append `next_token` and set `stepsLeft = new_steps`. Return only the
corrected Python body with the required rationale block.
"""

EVALUATION_FAILURE_REFINEMENT_PROMPT = """\
The previous strategy verified and ran, but evaluation was below threshold.

Previous strategy:
{previous_strategy}

Evaluation feedback:
{evaluation_feedback}

Improve the control policy using the current natural-delimiter helpers. For GSM, prefer
ordinary reasoning with `AppendUnconstrainedStep`, then natural answer opening
with `AppendUnconstrainedNudgeLeftDelimiterStep` or raw
`UnconstrainedNudgeLeftDelimiterStep`, then
`AppendConstrainedOrRightDelimiterStep` while
`helpers.IsComplete(generated) or helpers.CanConstrain(generated)`.
Avoid opening on the first local calculation, but also do not wait until only a
tiny budget remains before nudging for `<<`; natural delimiter opening needs
several attempts. Never put a
`not helpers.CanConstrain(generated): break` branch before the
`helpers.IsComplete(generated)` close branch; completion is the moment to allow
`>>`, not to exit. Return only the Python body with the
required rationale block.
"""

FORMAT_REPAIR_PROMPT = """\
Your output must be a Python method body. It is missing the required rationale
block markers.

Previous strategy:
{previous_strategy}

Rewrite it with:
# CSD_RATIONALE_BEGIN
# concise rationale
# CSD_RATIONALE_END

Then include the executable strategy body. Do not change helper names to removed
APIs. Return only Python statements.
"""

STRUCTURE_REPAIR_PROMPT = """\
The previous strategy has a structural issue:
{issue}

Previous strategy:
{previous_strategy}

Rewrite the body using only the curated helper surface. Prefer:
- `AppendUnconstrainedStep`
- `AppendUnconstrainedNudgeLeftDelimiterStep`
- `AppendConstrainedOrRightDelimiterStep`
- `EndsWithLeftDelimiter` / `EndsWithRightDelimiter`
- `IsComplete`, `CanConstrain`, `ValidContinuationCount`,
  `ParserDistanceToComplete`, `MinStepsToComplete`

Do not use removed helpers: soft constrained, top-k constrained, budget-aware,
extend-constrained, rollback/salvage, repetition, or direct LM logit-shaping
helpers. Avoid checkpoint rollback unless the task-specific prompt explicitly
requires it. Return only the corrected Python body with the required rationale
block.
"""


def _natural_delimiter_user_reminder() -> str:
    if not _env_flag("CSD_REQUIRE_NATURAL_DELIMITERS"):
        return ""
    return """\

Natural-delimiter mode reminder:
- Do not use forced delimiter helpers for GSM.
- Use `AppendUnconstrainedStep` for ordinary reasoning.
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
  unconstrained token generation.
- Keep unconstrained lead-in short (typically 0-3 steps), then enter the SQL span.
- Inside the SQL span, prefer `AppendConstrainedOrRightDelimiterStep` so
  completion can close `>>` naturally; avoid constrained-only loops that never
  emit a right delimiter.
- Consider taking a checkpoint before deeper constrained expansion and use
  `RestoreIfDead(generated, checkpoint)` for bounded local recovery.
- If you ever use `AppendUnconstrainedStep(...)` in Spider single-span mode,
  pair it with checkpoint recovery (`Checkpoint` + `RestoreIfDead` or
  `RestoreCheckpoint`) so unconstrained detours can roll back to a
  grammar-valid SQL prefix.
- Once `helpers.EndsWithRightDelimiter(generated)` is true, stop immediately.
  Do not enter an after-phase that emits additional helper steps, and do not
  open a second `<< ... >>` span.
- Do not rely on natural LEFT-delimiter nudges for Spider in this mode.
- Do not continue long unconstrained narration after deciding to open the answer
  channel; the run is format-first.
"""


def build_initial_prompt(task_description: str) -> tuple[str, str]:
    user_prompt = INITIAL_GENERATION_PROMPT.format(task_description=task_description)
    user_prompt += _natural_delimiter_user_reminder()
    user_prompt += _scratch_span_preference_reminder()
    user_prompt += _spider_single_sql_span_reminder()
    return _compose_system_prompt(), user_prompt


def build_verification_error_prompt(previous_strategy: str, error_message: str) -> tuple[str, str]:
    user_prompt = VERIFICATION_ERROR_REFINEMENT_PROMPT.replace(
        "{previous_strategy}", previous_strategy
    ).replace("{error_message}", error_message)
    return _compose_system_prompt(), user_prompt


def build_runtime_error_prompt(previous_strategy: str, error_traceback: str) -> tuple[str, str]:
    user_prompt = RUNTIME_ERROR_REFINEMENT_PROMPT.format(
        previous_strategy=previous_strategy, error_traceback=error_traceback
    )
    return _compose_system_prompt(), user_prompt


def build_compilation_error_prompt(previous_strategy: str, error_message: str) -> tuple[str, str]:
    # Kept for backward compat; compilation no longer happens but this builder is still called
    user_prompt = EVALUATION_FAILURE_REFINEMENT_PROMPT.format(
        previous_strategy=previous_strategy,
        evaluation_feedback=f"(runtime error) {error_message}",
    )
    return _compose_system_prompt(), user_prompt


def build_format_repair_prompt(previous_strategy: str) -> tuple[str, str]:
    user_prompt = FORMAT_REPAIR_PROMPT.format(previous_strategy=previous_strategy)
    return _compose_system_prompt(), user_prompt


def build_evaluation_failure_prompt(previous_strategy: str, evaluation_feedback: str) -> tuple[str, str]:
    user_prompt = EVALUATION_FAILURE_REFINEMENT_PROMPT.format(
        previous_strategy=previous_strategy, evaluation_feedback=evaluation_feedback
    )
    user_prompt += _natural_delimiter_user_reminder()
    user_prompt += _scratch_span_preference_reminder()
    user_prompt += _spider_single_sql_span_reminder()
    return _compose_system_prompt(), user_prompt


def build_structure_repair_prompt(previous_strategy: str, issue: str) -> tuple[str, str]:
    user_prompt = STRUCTURE_REPAIR_PROMPT.format(
        previous_strategy=previous_strategy,
        issue=issue,
    )
    user_prompt += _natural_delimiter_user_reminder()
    user_prompt += _scratch_span_preference_reminder()
    user_prompt += _spider_single_sql_span_reminder()
    return _compose_system_prompt(), user_prompt
