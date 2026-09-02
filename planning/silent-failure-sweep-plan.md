# Sweep: failures disguised as results

**Date:** 2026-07-28 · **Branch:** `all-fixes-integration`

## The one-sentence version

Something breaks, the code catches the error, and instead of stopping it returns
a number that looks like a real measurement — so the broken thing gets debugged
as if it were the model.

## Why this is urgent, not tidy-up

```
    A missing file            A model that genuinely
    (the harness broke)       got everything wrong
            |                          |
            +------------+-------------+
                         |
                    accuracy = 0.0
                         |
              INDISTINGUISHABLE downstream
```

One instance of this cost weeks. This sweep found more — including one that
makes the score look BETTER than it is, and one that is still producing a fake
zero on Spider today.

---

## Correction first: the previous fix does not work

Reported last session as fixed. It is not.

```
  spider eval starts
        |
        v
  pooled worker path  --->  module missing  --->  handled correctly now  (the fix works)
        |
        v
  sequential path
        |
        v
  _setup_environment()   evaluator.py:1940
        |
        v
  import ...common.vllm_startup   <--- ALSO MISSING. never existed, any branch
        |
        v
  ModuleNotFoundError -> same broad except -> accuracy=0.0, syntax=0.0, n=0
```

`--eval-backend` defaults to `vllm` (`run_synthesis.py:162`), so this is the
normal path, not an edge case. The earlier fix moved the failure one step
deeper and was reported as a cure.

**Different from the pool:** the pool modules exist on another branch and could
be restored. `vllm_startup` has never existed in any commit. It has to be
written.

---

## The part that costs money

A failed evaluation is not just recorded. It is sent back to the AI as
feedback:

```
  missing python module
        |
        v
  eval_result.success = False
        |
        v
  feedback_loop.py:2004  ->  failed_at = EVALUATION
        |
        v
  "Refining based on evaluation error..."
        |
        v
  the strategy-writing model is asked to fix it        <-- it cannot
        |
        v
  new strategy -> same missing module -> repeat
```

So the pipeline spends real API budget asking a model to fix a missing file,
over and over, and labels the result "the strategy failed evaluation".

---

## What the sweep found

116 wide error-catches were reviewed across five areas. Most are harmless. These
are not.

| # | Where | Break it and you get | Direction | Status |
|---|---|---|---|---|
| 1 | `evaluator.py:1940` | Fake 0% on the default path, today | too low | **fixed** |
| 2 | `feedback_loop.py:2004` | Broken harness blamed on the strategy | wrong target | **fixed** |
| 3 | `evaluator.py:2422` | Broken grammar file -> **syntax rate 100%** | **too high** | **fixed** |
| 3b | `grammar_decoder.py:160,182` | Decodes with NO grammar applied, unrecorded | **too high** | **fixed** |
| 4 | `parser_utils.py:408` | Mask lookup fails -> model forced to stop mid-query | too low | open |
| 5 | `model_utils.py:643` | Completeness check fails -> stopping always allowed | too low | open |
| 6 | `executor.py:259` | Missing database file -> correct SQL scored wrong | too low | open |
| 6b | `sql_spider_eval/evaluation.py:525,645` | Live Spider scorer; `eval_err_num` discarded | too low | open |
| 7 | `evaluator.py:1514` | Reference formula errors -> model marked correct | **too high** | **kept on purpose** |
| 7a | `evaluator.py:1577` | `int(` and `round(` treated as the same function | **too high** | **fixed** |
| 7b | `evaluator.py:2336` | z3 not installed -> every GSM answer scored wrong | too low | **fixed** |
| 7c | `eval_logic.py:171` | Unreadable field -> a *different grader* runs, silently | changes the method | **fixed** |
| 7d | `eval_logic.py:192` | A numeric grader CRANE does not have, averaged in | not comparable | **deleted** |
| 10 | `evaluator.py:2561` | Blanket catch ate every fix above | too low | **fixed** |
| 10b | `run_reference_strategy.py:143` | Broken harness saved as `accuracy: 0.000` | too low | **fixed** |
| 11 | `model_utils.py:1903` + runner default | `--device auto` cannot start vLLM, so **the default flags always fail** and the failure is saved as a 0% score | too low | **found by running it** |
| 8 | `metrics.py:79` | RDKit crash -> "invalid molecule" | too low | open |
| 9 | `dataset.py:148` | Corrupt file -> example vanishes from a fixed split | changes the set | open |

### 3b was the worst, and it was found last

`SyncodeLogitsProcessor.__call__` has two paths that skip the masking step
entirely — a parser error, and "no acceptable tokens". Either one means those
tokens were chosen with **no grammar applied at all**.

The only record was one bool, `parse_failed`: written in 5 places, read in
exactly one — its own print guard. Nothing outside the class ever saw it, and
the second path never even set it.

Unconstrained decoding often still produces valid output, so a run could abandon
the grammar and still report a high validity rate, which reads as evidence that
constrained decoding works. Now counted per cause, and the count survives the
per-sample resets that used to clear the flag.

Found only because the vendored `syncode/` tree was NOT waved through as
third-party. Skipping it would have been the same mistake in miniature —
assuming a category is safe instead of checking.

### #3 is the dangerous one

The others make results look worse than reality, which is annoying but gets
noticed. #3 makes them look **perfect**:

```python
except Exception:
    return True, [(m, True) for m in matches]   # every segment "valid"
```

An unreadable grammar file means nothing is ever checked, and the run reports
100% syntactic validity. Unchanged since the file's first commit.

### #4 quietly undoes a fix we already made

```
  mask lookup throws
        |
        v
  returns a mask with nothing allowed        parser_utils.py:408
        |
        v
  "nothing else is legal" == a genuine dead end
        |
        v
  dead end is exactly when stopping early is PERMITTED
        |
        v
  end-of-text becomes the only option        model_utils.py:1158
        |
        v
  the "no valid tokens" alarm cannot fire — the fake
  end-of-text is what makes the count non-zero
```

The previous session added a rule that the model may only stop when its query
is complete. This path walks around it.

---

## The fix, in order

Each step gets a test that fails first.

```
  STEP 1   write the missing vllm_startup helpers
           -> the default eval path can start at all

  STEP 2   separate "the harness broke" from "the strategy was bad"
           -> a missing module stops the run instead of being sent
              to the AI as feedback
           -> THIS IS THE ONE THAT CATCHES THE WHOLE CLASS

  STEP 3   #3 and #7 (things that inflate the score)

  STEP 4   #4, #5, #6 (things that deflate it)

  STEP 5   #8, #9, and delete the dead split_provenance import

  STEP 6   a guard test: every module the repo imports must exist
           -> catches the next lost file before it becomes a score
```

Step 2 is the real fix. Steps 3–5 are instances.

---

## #7 — decided: comparability wins

The label was checked against the real thing, not assumed. Upstream is
`src/prompting/gsm_symbolic.py` in `github.com/uiuc-focal-lab/CRANE`, and the
asymmetry is genuinely theirs, at lines 114-121:

```python
except:  return False    # the MODEL's formula throws  -> not correct
except:  return True     # the GOLD's formula throws   -> CORRECT
```

Their line 204 confirms which is which: `LLM expression {expr1}`, `GT
expression {expr2}`.

**Aadivya's call: match CRANE exactly, including where CRANE is wrong.** A
grader that is better than theirs produces a number that cannot be compared to
their published one, which hurts the paper more than being faithfully wrong.

So the rule above stays. Three of our *deviations* from it were corrected — we
had quietly improved on CRANE in three places:

```
  ours was STRICTER than CRANE          ours was LOOSER than CRANE
  (undercounts vs their number)         (overcounts vs their number)
            |                                     |
  untyped variable -> refused           round( repaired -> int( and
  to sample at all                      round( became one function
            |                                     |
            +------------------+------------------+
                               |
                    both now match upstream
```

### The `round(` one was a real score inflater

Upstream writes `re.sub(r'\round\(', 'ToInt(', expr2)`. The `\r` is the
carriage-return escape, so the pattern is CR + `ound(` and never matches. Their
conversion is dead code, the gold keeps its `round(`, and grading falls through
to the sampler — where a plain `eval` computes `round()` correctly.

Someone here repaired the typo. That is not a fix: `ToInt` truncates and
`round` rounds. Converting both `int(` and `round(` to `ToInt(` makes them the
same function, so a model answering `int(x/3)` was *proved* equal to a gold of
`round(x/3)`. The regex is now deliberately dead again, with a comment saying
why, so nobody repairs it a third time.

### Two guards CRANE lacks were removed, and here parity agreed with honesty

Upstream calls plain `eval()` on `variable_types` and lets a bad value stop the
run. Ours caught it. Both catches are gone:

```
  z3 not installed
        |
        v
  ModuleNotFoundError -> except Exception -> return False
        |
        v
  EVERY GSM answer scored wrong -> accuracy 0.0
```

Measured before the fix, not inferred: `a * 2` against a gold of `a + a` — the
same formula — graded **False** on this machine, purely because z3 is absent.
That is finding #1 all over again, in a different file.

### #7c was the one that would have survived the fix

`eval_logic.py` had the same `except -> {}` one level up, and `{}` is falsy:

```
  variable_types unreadable
        |
        v
  vt = {}          <-- looks exactly like "this row never had one"
        |
        v
  `if vt and ...` is False -> symbolic grader never called
        |
        v
  scored by NUMERIC comparison instead, averaged in with the rest,
  nothing anywhere records that the method changed
```

Not a wrong number — a *different measurement* mixed into the same average.
Fixing only the inner function would have changed nothing on this path, because
the caller had already turned the failure into a routing decision. Same wiring
gap as step 2; found by checking the caller rather than trusting the fix.

**Left alone on purpose:** the anti-hang guard on runaway expressions stays (it
has no upstream counterpart, but removing it risks a hang native code cannot be
interrupted from). A known, deliberate parity break.

### #7d — the numeric grader is gone, not just bypassed

The first version of #7c kept the numeric path for rows that genuinely had no
`variable_types`, calling it a real decision rather than a swallowed error.
That was wrong. CRANE has **no numeric grader at all** — its `parse_answer`
(gsm_symbolic.py:28-56) either proves the two formulas equivalent or leaves
`correct = False`. Every example our fallback graded was measured a way CRANE
does not measure, then averaged into a figure reported against CRANE's number.

It was not a rare path either: `answer_parsed` defaults to `''` (dataset.py:65
and :164), which is falsy, so any source row missing that field took it.

Both routes into it now raise `UngradableExample`. Nine methods deleted rather
than left dormant behind a branch nobody takes — `_extract_answer_gsm`,
`_answers_match`, and the seven helpers underneath them that nothing else
reached. `_truncate_gsm_output` survives; it has five other callers.

---

## #10 — the catch that was eating all of the above

Found by a judge agent, not by me. Every fix in #7b, #7c and #7d raises. They
all landed here:

```
  _evaluate_one_example()                        evaluator.py:2561
        |
        v
  except Exception  ->  sample{is_correct: False, accuracy_applicable: True}
        |
        v
  counted in num_accuracy_examples (the accuracy denominator)
        |
        v
  a missing module and a wrong answer are the same number again
```

The docstring even promised it: *"It never raises"*. Three fixes were dead on
arrival — the identical wiring gap as step 2 and #7c, three times in one
session. The lesson is now the rule: **after fixing an inner function, read the
caller before claiming it works.**

Three names re-raise, and only three:

```
  ModuleNotFoundError / ImportError   the harness broke; nothing was measured
  UngradableExample                   the row has no field to grade against
```

Everything else stays caught. A model producing nonsense, a timeout, a solver
falling over on one expression — those are real outcomes, and aborting a whole
run over one of them loses every other example.

**Why a named class instead of `TypeError`/`ValueError`:** those are what any
ordinary bug raises (a `None` where a string was expected, `float("abc")`).
Re-raising them by type would turn one stray bug in the generation path into an
aborted evaluation run. `UngradableExample` says the one thing meant and
nothing else; it subclasses `ValueError` so existing handlers are unaffected.

### #10b — and the same bug one file over

Checking who consumes the re-raise turned up a fresh instance.
`feedback_loop.py` handles it correctly: `classify_eval_failure` (:303-322)
keys off `num_examples`, returns `HARNESS`, and :2038-2052 stops the run with
the real error instead of feeding it to the strategy model.

`run_reference_strategy.py` did not. `_evaluate` read `result.accuracy` (:143)
with no check, and `main()` wrote it to `--output-json` and printed
`accuracy=0.000` (:219-224). `error` never reached the payload, so the saved
file claimed a measured zero with nothing to contradict it. Now guarded the way
`reevaluate_compiled_csd.py:94-97` already did.

**Checked and left alone:** `baseline_store.py:54-96` has no guard either, but
its only caller (`reevaluate_compiled_csd.py:106`) checks first, so it is latent
rather than live. Noted, not fixed.

---

## What will still be unproven afterwards

Every item above is verified by reading the code and tracing who consumes the
value. None of it proves Spider scores above zero. That needs a real run on
`focal` reporting `num_examples > 0`.

The check that matters is not "the tests pass". It is "an example actually ran".

---

## 2026-07-29: caught in the act on focal

A 5-example Spider run on focal's own code, unchanged, produced this:

```
[sharded-eval] worker 0 FAILED mid-shard: RuntimeError('vLLM runtime currently
               requires a CUDA device in this project.')
[sharded-eval] all workers dead; falling back to in-process eval for the remainder
Wrote outputs/probe/spider_unconstrained_n5.json: accuracy=0.000 syntax_rate=0.000
```

and wrote this file:

```json
{ "accuracy": 0.0, "syntax_rate": 0.0, "num_examples": 0, "answers": [] }
```

Every worker died, the in-process fallback produced nothing, and it still saved
`accuracy: 0.0` with nothing in the file saying anything had gone wrong. So the
half of the question that was open is now settled by observation rather than by
reading: **Spider's 0% is a broken harness being written down as a score.**
Item 10b stops exactly this, and it is not on focal.

Still open: whether Spider scores above zero once the harness runs. That needs
this same run to reach `num_examples > 0`.

### Item 11, found by running it rather than reading it

The crash above is its own instance of the pattern, one layer further out.
`create_vllm_lm` (`model_utils.py:1903`) raises unless the device name starts
with `cuda`, and the runner's default is `--device auto`. `"auto"` does not
start with `"cuda"`, so **the default flags cannot succeed** — and the failure
does not surface as "you passed a bad flag", it surfaces as a benchmark score of
zero.

That is worth separating from the rest of the list. Items 1-10b are all a
*failure* being recorded as a measurement. Item 11 is a *configuration mistake*
being recorded as a measurement, which is worse in one specific way: nobody
suspects the flags when the number looks like a plausible bad result.

The launch that actually works, for the record:

```
CUDA_VISIBLE_DEVICES=<a free gpu> python -m synthesis.evaluate.run_reference_strategy \
  --strategy unconstrained --dataset spider \
  --eval-model Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --device cuda \                      # NOT the default "auto"
  --vllm-gpu-memory-utilization 0.35 \ # focal is shared; 0.6 loses to other jobs
  --vllm-max-model-len 4096 \
  --eval-sample-size 5 --eval-max-steps 300 \
  --output-json outputs/probe/spider_unconstrained_n5e.json
```

Two things about that command are not obvious and cost several attempts each:

- `--device cuda` and `CUDA_VISIBLE_DEVICES` are both needed. The first gets
  past the string check; the second points the run at a card with free memory,
  because the worker pool otherwise lands on `cuda:0` and asks for more than is
  free there.
- The first run on a new tokenizer spends ~25 minutes building a grammar mask
  store, then caches it under `cache/mask_stores/`. Later runs skip it. Budget
  for that once, not every time.
