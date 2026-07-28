# Merging focal's tree into the local tree

**Date:** 2026-07-29
**Worktree:** `/Users/aadivyar/Documents/Research/Dynamic CSD Gen/local-finalization/csd-merge`
**Branch:** `merge-focal-into-mine` (from `all-fixes-integration`, `80b44160`)
**Merging in:** `origin/synthesis-snapshot-20260622` (`a9e96f73`)
**Status:** DONE — all 13 conflicted files resolved, merge committed as `ffae39ff`.

## Why

The two trees had diverged: 17 local commits vs 92 on focal. Aadivya's call was
"merge with focal — our local version is just a newer version of what focal is
doing." The resolution rule agreed up front:

> **focal wins on how things run; the local tree wins on what gets recorded as a result.**

"How things run" = GPU sizing, vLLM startup, engine placement, prompt rendering.
"What gets recorded" = anything that decides a number a human reads.

## Resolved (11 files, all parse clean, zero conflict markers)

| File | Resolution |
|---|---|
| `synthesis/evaluate/evaluator.py` | Mine on all 10 hunks — this file *is* the Spider bug |
| `synthesis/evaluate/feedback_loop.py` | Composed; see "restart machinery" below |
| `synthesis/run_synthesis.py` | Both sides' helpers kept; both argparse flags reconstructed |
| `synthesis/run_constants.py` | focal's GPU memory table; mine on the retired delimiter table |
| `synthesis/generate/generator.py` | Composed — focal's setter plus my flag assignment |
| `synthesis/evaluate/benchmarks/common/model_utils.py` | Both sides' stop rules kept, one composed branch |
| `synthesis/evaluate/benchmarks/common/parser_utils.py` | focal wholesale (see cost below) |
| `synthesis/evaluate/benchmarks/common/vllm_startup.py` | Mine, plus two of focal's three memory-error strings |
| `synthesis/evaluate/benchmarks/smiles/eval_logic.py` | Mine on the prompt (CARS parity) |
| `synthesis/evaluate/benchmarks/gsm_symbolic/prompts.py` | Mine (byte-parity with CRANE) |
| `synthesis/tests/fairness_cloud/test_helper_menu_prune.py` | Mine, `EXPECTED_UNIVERSE_SIZE = 65`, verified by running it |
| `synthesis/evaluate/evaluator.py.bak_uvmetric_20260610` | Deleted (stray backup) |

### The hunk that mattered most

`evaluator.py`, in `evaluate()`. focal imports the worker pool *inline inside the
`try`*:

```python
from synthesis.scripts.eval_worker_pool import get_synthesis_eval_pool
```

If that module is missing, `ModuleNotFoundError` falls through to the
`except Exception` at the bottom of the method, which returns
`accuracy=0.0, syntax_rate=0.0, num_examples=0`. **That is exactly how Spider
read 0% for weeks.** The merged file uses `_resolve_eval_pool_loader()` instead,
which catches the import failure, says so on stdout, and falls back to the
slower sequential path — so a missing file costs speed, not a fake score.

Also kept from the local tree: `_evaluate_one_example` re-raises
`ModuleNotFoundError`, `ImportError`, and `UngradableExample` instead of scoring
them as wrong answers. That list is fixed at three names and is guarded by
`tests/test_harness_failures_are_not_scored_as_wrong_answers.py`.

### Restart machinery (feedback_loop.py)

focal's rewrite deleted machinery the local restart branches still call. After
resolving the 9 marked conflicts the file parsed but a dependency sweep found
`use_restart` referenced twice with no assignment and `_apply_restart_cooldown`
called twice with no definition. Restored from `git show HEAD:`:

- `_should_restart()` and `_apply_restart_cooldown()` methods
- the two constructor assignments
- the two `use_restart = self._should_restart(attempts)` call sites

Also restored: the `save_reports` constructor parameter. focal dropped it, but
`run_synthesis.py:813` and `tests/test_attempt_cap_and_incremental_save.py`
(lines 186, 283) both still pass it — without the parameter they fail with a
`TypeError` on the first call.

### Two judgement calls worth revisiting

1. **`parser_utils.py` — a caching optimisation was dropped.** The local side
   cached prefix→text on the prefix alone. focal introduces *two* texts per
   prefix (`_structured_text` prepends the `<<` opener for CRANE,
   `_complete_text` does not), which makes a prefix-only cache key wrong, and
   focal also adds real CARS llguidance `try_consume_tokens` logic. Took focal
   wholesale and verified no orphaned cache references remain. **Cost: lost
   speed, not lost correctness.**

2. **`vllm_startup.py` — added two of focal's three memory-error strings.**
   focal's nested `_is_vllm_startup_memory_error` matched three substrings; the
   local module-level version matched none of them. Added
   `"desired gpu memory utilization"` and `"free memory on device"` — both are
   real vLLM out-of-memory wording, and missing them meant the retry ladder
   refused to back off on a busy shared GPU. Deliberately did **not** add
   `"Engine core initialization failed"`: that is vLLM's generic wrapper around
   *any* startup failure, so treating it as memory pressure would send a missing
   module around the retry ladder and then report it as a generic startup
   failure — the same laundering pattern this whole sweep is about.

## The two reference strategies (`.dfy`) — both kept local

These define what the baselines *are*, so changing them changes every
comparison number. Both went to the local version.

- **itergen** — Aadivya's call. The two sides are different algorithms, not two
  versions of one. focal steps token-by-token via `SafeSoftConstrainedStep`; the
  local version makes one `RegenerateUnitOnGroundingFailure` call that does the
  whole constrained phase with rollback and penalty. The automatic merge had
  spliced focal's last line into the local `else` branch where its `next`
  variable does not exist, so that splice would not have compiled — it had to be
  one side or the other.

- **cars** — decided by measurement, see the next section.

All six helpers the kept `cars.dfy` calls (`AppendConstrainedToken`,
`CloseConstrainedSpan`, `ConstrainedStep`, `SafePenalizedConstrainedStep`,
`SoftConstrainedStep`, `UnconstrainedStep`) exist exactly once in the merged
`synthesis/verify/library/VerifiedAgentSynthesis.dfy`.

## Measurement: focal's CARS stop rule is dead code, and dangerous when it isn't

**Question.** focal's `cars.dfy` leaves the decode loop at:

```dafny
if inside && parser.IsCompletePrefix(cur) && parser.ValidNextTokenCount(cur) == 0 {
```

with the comment *"bare `C` can be complete yet still extendable"* — i.e. the
`ValidNextTokenCount == 0` half was added specifically to stop the collapse that
produced 35/50 `<<C>>` answers. Does it fire at `"C"`?

**How it was measured.** No model and no GPU — this is a pure grammar question.
Script: `saved-results/scripts/cars_stop_rule_probe.py`, run on focal, **19
seconds wall clock**. It loads the Qwen2.5-Coder-1.5B tokenizer, reads the
already-cached grammar mask store, builds the SMILES parser exactly the way
`smiles/eval_logic.py:78-85` does (`accept_mask_backend="llguidance"`), and
evaluates both predicates directly.

**Result** (`smiles_chain_extenders.lark`, vocabulary 151,665):

| prefix | `IsCompletePrefix` | `ValidNextTokenCount` | break fires? |
|---|---|---|---|
| `''` | False | 463 | no |
| `'C'` | True | 22889 | no |
| `'CC'` | True | 22889 | no |
| `'CCO'` | True | 22489 | no |
| `'OCCO'` | True | 22489 | no |
| `'C1'` | True | 20384 | no |
| `'C1CCCCC1'` | True | 20384 | no |
| `'N'` | True | 22972 | no |

**It never fires.** And it cannot: the grammar's `smiles: first_term rest*`
allows another atom after any valid molecule, so a valid prefix always has
continuations. The guard is dead code — focal's loop can only end on eos, or on
a rejection that throws the whole span away.

**But there is one way to make it fire, and it is a bug.** `parser_utils.py:413`
ends the accept-mask computation with:

```python
                except Exception:
                    # Fallback on parse error
                    import torch
                    return torch.zeros(len(self._token_list), dtype=torch.bool)
```

An all-zero mask means "no token is allowed", so `ValidNextTokenCount` returns 0
and focal's break fires. Any crash inside the mask code therefore reads to the
strategy as *"the grammar says this molecule is finished"* — the same laundering
pattern as the Spider bug, one layer down.

This is not hypothetical: the first version of the probe passed an 18-token
vocabulary, which made the unguarded `accept_mask & self._forbidden_allow_mask`
on line 411 throw on a size mismatch, and the probe reported `ValidNextTokenCount
= 0` and *"break fires"* at every single prefix. A crash produced a clean,
plausible, completely wrong table. The llguidance branch on line 364 guards the
same `&` with a size check; the mask-store branch does not.

**Decision.** Kept the local `cars.dfy`, which ends the span on an explicit stop
signal over a complete molecule (`closeRequested` / `CloseConstrainedSpan`)
rather than on a token count that is either always non-zero or reporting a
crash.

**Not settled by this probe:** focal's `CarsTrieStep` / `RejectLastInTrieHelper`
are backed by a real trie in `model_utils.py:1072`, whereas the local version
models the rejection inline with a `rejectedTokens` list that resets each
invocation. Which of those is the more faithful CARS is a separate question and
was not tested here.

## Measurement: what the dropped `parser_utils.py` cache was worth

Same harness, `saved-results/scripts/parser_cache_bench.py`, run on focal.

Every grammar question (`IsValidPrefix`, `IsCompletePrefix`,
`ValidNextTokenCount`, `ValidNextToken`, `GroupHasValidMember` — about five per
decode step) needs the answer-so-far as plain text, but it is stored as a list of
token objects. Converting the list walks every token. The dropped cache stored
that text so calls 2-5 got it free.

| answer length | one conversion | per decode step | per example | per 50-example eval |
|---|---|---|---|---|
| 50 tokens | 81 µs | 0.41 ms | 0.010 s | 0.5 s |
| 100 | 157 µs | 0.78 ms | 0.039 s | 2.0 s |
| 200 | 314 µs | 1.57 ms | 0.157 s | 7.8 s |
| 400 | 626 µs | 3.13 ms | 0.626 s | 31 s |
| 800 | 1254 µs | 6.27 ms | 2.51 s | 125 s |

A cache hit costs **0.36 µs**, so at a 100-token answer the cache is ~440×
cheaper than redoing the conversion. Cost grows with the square of answer length
(the answer gets longer as you generate, and every step re-converts the whole
thing).

**Why it was dropped, and how to restore it.** The old cache keyed on the token
list alone. focal's version derives *two* different texts from the same list —
`_structured_text` prepends the `<<` span opener for CRANE, `_complete_text`
does not — so one key had two correct answers and the cache would return the
wrong string. The fix is to widen the key from `(id(prefix), len(prefix))` to
`(id(prefix), len(prefix), which_text)`. Nothing else about the old cache was
wrong.

**Worth doing where answers are long (Spider, GSM), near-worthless for SMILES**,
where a chain extender is 10-30 tokens and the whole table above rounds to zero.

## Related finding, not yet chased

Every recorded CARS result under `outputs/controlled_comparison*/` has
`num_examples` absent from the JSON. The "check `num_examples` before believing
`accuracy`" test therefore **cannot be applied to any recorded CARS number**.
One of them, `outputs/controlled_comparison/spider_7B/cars.json`, reads
`accuracy=0.06, syntax_rate=0.07` — the same near-zero shape as the Spider bug,
and there is currently no way to tell from the file whether it is real.

## How to resume

```bash
cd "/Users/aadivyar/Documents/Research/Dynamic CSD Gen/local-finalization/csd-merge"
git diff --name-only --diff-filter=U     # the two .dfy files
# resolve them, then:
git add synthesis/verify/reference/cars.dfy synthesis/verify/reference/itergen.dfy
git commit                                # do NOT use `git add -A` in this repo
```

**Never `git add -A` or `git add .` here** — nothing is git-ignored and it
sweeps in a 64 MB `.nltk_data/` cache. Use `git add -u` plus explicit paths.

To redo a botched resolution on one file:
`git checkout --merge <path>` recreates the conflict markers.

## Repo size warning

The merge stages 27,901 files, of which ~24,800 are focal's committed
`outputs/` — including run logs of 97 MB, 92 MB, 88 MB, 75 MB, 73 MB, 65 MB,
62 MB, and 60 MB. These blobs already exist in focal's history, so merging does
not make the repo bigger than focal already is, but the local branch gains them.
