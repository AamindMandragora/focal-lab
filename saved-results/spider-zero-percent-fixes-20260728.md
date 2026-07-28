# Fixing Spider's 0% accuracy / 0% syntax

**Date:** 2026-07-28
**Branch:** `all-fixes-integration` (in worktree `csd-integration`), built from three
separate branches merged together.
**What this is for:** Spider (text-to-SQL) was scoring 0% on both accuracy and
syntactic validity. This records what was actually wrong, what was changed, what
was checked, and what is still open.

---

## The short version

**Read the root cause section first.** The 0% was not caused by any of the four
faults originally investigated. It was caused by a missing file. The other fixes
are real improvements, but none of them could have moved the score, because not
a single Spider example was ever evaluated.

| # | Fault | Status |
|---|---|---|
| **0** | **A missing file made every Spider eval report a fake 0%** | **Fixed — this was the root cause** |
| 1 | Decoding could stop in the middle of an unfinished query | Fixed (Python + Dafny) |
| 3 | The strategy-writing AI was never told which decoding surface it was on | Fixed |
| 4 | One slow attempt could hang a run; results only saved at the end | Fixed, now settable from the CLI |
| 5 | Strip a leading newline before the constrained region | **Deliberately not done** |
| — | SMILES said it had no delimiters but asked the model for them | Fixed |

---

## Fault 0 — the root cause: a missing file reported as a 0% score

This is the one that actually explains the symptom.

```
POOLABLE_DATASETS = {"gsm_symbolic", "spider"}          evaluator.py:156
        |
        |  spider is in the set, so evaluation takes the pooled branch
        v
from synthesis.scripts.eval_worker_pool import ...      evaluator.py:3033
        |
        |  that module DOES NOT EXIST on this branch
        v
ModuleNotFoundError  (a subclass of Exception)
        |
        |  caught by the broad `except Exception` wrapping the whole method
        v
return EvaluationResult(accuracy=0.0, syntax_rate=0.0,  evaluator.py:3062
                        num_examples=0, success=False)
```

So **a missing file was displayed as "the model got everything wrong"**, on every
iteration, no matter what the strategy did. Nothing downstream could tell the two
apart. `num_examples=0` was the tell: a genuine 0% run still evaluates its
examples.

Three files are absent — `eval_worker_pool.py`, `sharded_eval_core.py`,
`eval_worker_main.py`. Verified with `git ls-tree`: absent from this branch, from
`relaunch-spider-queue`, and from `origin/main`; present on
`codex/verification-burden-reduction-20260725`. Likely lost in the big-file
history rewrite that also dropped the `prompt_rendering` package.

**The fix:** the pool is only a speed optimisation, and a working sequential path
already exists in the same method. Choosing between them now goes through
`_resolve_eval_pool_loader()`, which returns `None` instead of raising, and says
loudly which path it took. A missing optimisation now costs time, not
correctness.

```
[sharded-eval] eval worker pool unavailable (ModuleNotFoundError: No module
named 'synthesis.scripts.eval_worker_pool'); falling back to the slower
sequential eval path.
```

**The wider lesson:** a broad `except Exception` that returns a plausible-looking
score is worse than a crash. A crash gets investigated; a 0% gets debugged for
days in the wrong place. That is exactly what happened here.

**Speed cost, still open:** Spider now reloads the vLLM engine (~24s) every
iteration. Restoring the three pool modules would win that back, but they were
never on this branch and would need GPU testing on `focal` — recommended as a
follow-up, not done blind.

---

## Fault 1 — stopping before the answer is finished

**Was:** the end-of-text token was added to the allowed set unconditionally. The
model could therefore end its turn part-way through a query, which is both wrong
and unparseable — that alone can produce 0% syntax.

**Now:** end-of-text is allowed only when the text so far is a complete, parseable
query, or when nothing else is legal (so decoding can never deadlock).

`synthesis/evaluate/benchmarks/common/model_utils.py:631`

```python
def _eos_is_legal(parser, prefix, has_other_valid_token: bool) -> bool:
    if not hasattr(parser, "IsCompletePrefix"):
        return True          # parser can't tell us -> old behaviour
    try:
        is_complete = bool(parser.IsCompletePrefix(prefix))
    except Exception:
        return True
    return is_complete or not has_other_valid_token
```

Applied at both call sites: `MaskValidNextAndEos` (~1157) and
`BoostValidNextAndEos` (~1186).

The same rule was written into the Dafny library as a postcondition on 16
methods, so the proof and the running code state the same guarantee:

```dafny
ensures next == eosToken ==>
        (parser.IsCompletePrefix(prefix) || parser.ValidNextTokenCount(prefix) == 0)
```

`TopValidCandidates` was rewritten to *derive* this from `MaskValidNextAndEos`
rather than assume it, and its unreachable "if the pool is empty, fall back to
end-of-text" branch was deleted — falling back to end-of-text is precisely the
bug being removed.

---

## Fault 3 — the author didn't know which surface it was writing for

There are two ways generation reaches the constrained region:

```
  visible-delimiter surface          observed surface
  --------------------------         ----------------------------
  starts OUTSIDE                     starts INSIDE
  strategy calls                     strategy calls
    OpenConstrainedSpan                EnterObservedConstrainedSpan
  which emits a literal "<<"         which emits nothing
                                     no "<<" EVER appears
```

Spider runs on the **observed** surface. The prompt never said so, so authors kept
writing strategies whose only route into constrained mode was *"wait until the
next token is `<<`"*. On Spider that token never arrives, so **nothing was ever
constrained** — a clean explanation for 0% syntax.

The prompt now states the surface. Because a comment is not a test, a guard test
(`test_decoding_surface_matches_reality.py`) runs the real generation runner and
checks that what the author is *told* matches what evaluation actually *does*,
under both settings of the mode switch.

---

## Fault 4 — attempt cap and saving as you go

A per-attempt time cap (default 3600s) marks an over-running attempt as timed out
and lets the loop continue instead of hanging; partial records are kept; results
are written to disk during the run rather than only at the end.

Four things were flagged for review here. Checking them found that **two of the
four flags were wrong** — worth recording, because they were stated with more
confidence than they had earned:

| Flag | Verdict |
|---|---|
| "the cap is scoped to the eval stage only" | **Wrong.** `attempt_start_time` is set at `feedback_loop.py:1705`, before compile and verify, and `attempt_elapsed` at :1967 measures the whole attempt. The real limit is narrower: the clock covers everything, but only evaluation can be *interrupted* part-way — Dafny verification runs to completion. Now stated in the flag's help text. |
| "a now-dead `eval_worker_pool` path" | **Badly wrong.** Not dead — live, reachable on Spider, and the root cause of the entire 0%. See Fault 0. |
| "no CLI flag in `run_synthesis.py`" | **Right about the gap, wrong about the file** — the entry point is `synthesis/run_synthesis.py`, not repo root. Fixed: `--max-attempt-seconds` added and forwarded; 0 or less means no cap. |
| "the 3600s default" | Kept. It is still a guess rather than a measured number, but it is no longer the only reachable value, which was the real problem. Worth re-checking now that eval runs sequentially and is slower. |

---

## Why fix 5 was deliberately NOT made

The proposal was to extend `remove_left_whitespace`
(`synthesis/evaluate/syncode/syncode/dfa_mask_store.py:419`) to strip a leading
**newline**, as it already strips a leading space.

That would have re-created fault 1 under a different spelling. In `sql.lark` a
newline is not whitespace — it *is* the end-of-query marker:

```lark
csd_start: sql_stmt EOQ
EOQ: /[;\n]+/
%ignore WS_INLINE      # spaces and tabs ONLY
```

So stripping a leading newline lets the model end the query before it has written
one — the identical failure to the end-of-text bug. The honest alternative, if
this ever needs revisiting, is to permit a leading newline *before* the
constrained region opens, outside the constrained span, rather than teaching the
mask store to swallow one.

---

## SMILES — made to match its own description

SMILES reported `emits_visible_delimiters() == False` and its grammar
(`start: smiles`) has no delimiter, yet all three prompt builders told the model
to wrap its answer in `<< >>`, and generation started outside the constrained
region. Code said one thing, did another. Per decision on 2026-07-28 the
behaviour was made to match the declaration: prompts no longer mention
delimiters, and generation starts inside the constrained region.

Safe to do because extraction (`clean_smiles_output`, `metrics.py:44`) *strips*
`<<`/`>>` rather than requiring them.

---

## What merging the three branches exposed

This is the part worth remembering: **each branch passed its own tests, and the
combination still failed.**

1. `_start_inside_constrained()` looked the benchmark up with `get_logic()`,
   which **raises** `ValueError` for an unregistered dataset. That value only
   chooses a sentence of wording in the author's prompt, so a wrong guess should
   cost one weaker strategy — never the whole run. Its docstring already promised
   a fallback; now it delivers one.
   (`synthesis/evaluate/feedback_loop.py:778`)
2. The attempt-cap tests' stand-in generator had drifted from the real
   `StrategyGenerator`, which fix 3 had given a new `start_inside_constrained`
   argument. Signatures brought back in step.

---

## Verification actually run (not claimed)

```
Dafny:  /opt/homebrew/bin/dafny verify synthesis/verify/library/VerifiedAgentSynthesis.dfy
        -> Dafny program verifier finished with 179 verified, 0 errors

Tests:  32 passed        (eight files, run together, in the merged tree)
```

Checked that the Dafny result was **not** faked: no `requires`/`ensures` were
removed or weakened, and no `assume` / `{:axiom}` / `{:verify false}` were added.

Merge of the three branches produced **zero conflicts**.
Total change from the fork point: 16 files, +1430 / −52.

### How to re-run

```bash
cd ".../local-finalization/csd-integration"
VP=".../local-finalization/csd-generation/.venv/bin/python"   # system python has no torch

"$VP" -m pytest \
  tests/test_eos_requires_complete_prefix.py \
  tests/test_author_prompt_states_decoding_surface.py \
  tests/test_decoding_surface_matches_reality.py \
  tests/test_smiles_has_no_delimiters.py \
  tests/test_attempt_cap_and_incremental_save.py \
  tests/test_surface_lookup_never_kills_a_run.py -q

cd synthesis/verify/library && /opt/homebrew/bin/dafny verify VerifiedAgentSynthesis.dfy
```

Notes: `rtk proxy <cmd>` if a bare command fails to spawn. Running the whole
`tests/` directory fails on `tests/grounding_symbol_boundary/` because `syncode`
isn't installed in that venv — pre-existing, unrelated to this work.

**Never `git add -A` in this repo** — nothing is git-ignored and it sweeps in a
64 MB cache. Use `git add -u` plus explicit paths.

---

## Correction (2026-07-28, later the same day)

**The claim below that "the cause of the fake 0% is removed" was wrong when
written.** Recorded here rather than edited away, because the mistake is the
same one this document is about.

The Fault 0 fix routed evaluation away from the missing worker-pool modules and
onto the sequential path. That path calls `_setup_environment`
(`evaluator.py:1940`), which imports
`synthesis.evaluate.benchmarks.common.vllm_startup` — a module that **has never
existed in any commit on any branch**. `--eval-backend` defaults to `vllm`
(`run_synthesis.py:162`), so this is the normal path, not an edge case. The
`ModuleNotFoundError` landed in the same kind of broad `except` and produced the
identical `accuracy=0.0, syntax_rate=0.0, num_examples=0`.

So the failure was moved one level deeper and reported as a cure. The check this
very document recommends — prove `num_examples > 0` before believing anything —
is the check that was not run against the fix itself.

`vllm_startup.py` has since been written and is covered by
`tests/test_vllm_startup_helpers.py`. That still does not prove Spider scores
above zero; see "Still open".

---

## Still open

- These fixes are verified by tests and by Dafny. **They have not yet been shown
  to move the actual Spider score off 0%** — that needs a real evaluation run on
  the GPU box (`focal`, needs the campus VPN). Until then the honest claim is
  "several real defects are fixed and two fake-zero causes are removed", not
  "Spider now scores X", and not "the cause is removed" — that was said once
  already and was wrong.
- **Restore the three pooled-eval modules?** Doing so would recover the ~24s
  per-iteration engine reload. They exist on
  `codex/verification-burden-reduction-20260725`. Not done here because they were
  never on this branch and cannot be tested without GPUs — needs a run on focal.
- **Hunt the rest of this bug class.** DONE — 116 wide catches reviewed across
  five areas. Findings and current status are in
  `planning/silent-failure-sweep-plan.md`. Four are fixed; several remain open,
  and two need a decision from Aadivya (delete two dead files; whether the
  GSM reference-formula shortcut is deliberate CRANE parity).
- The 3600s cap default is still a guess; re-check it now that eval runs
  sequentially and is therefore slower.
