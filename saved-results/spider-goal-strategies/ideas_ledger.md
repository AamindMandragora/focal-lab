# Ideas Ledger — Spider cycle-1 cold runs

**Purpose:** a running list of ideas / hypotheses / things-to-try that occur to me while watching
the runs. Each entry gets a **testable prediction** so it can be scored later. When the runs finish
(ALL DONE / exit=) **or** we cancel them, I do a **post-run validation pass** at the bottom: for each
idea, was it borne out by how the run actually went? This separates "sounded smart in the moment"
from "actually mattered."

**ACTIVE runs (relaunched 2026-06-23 ~12:43 UTC, the 50-example fast-iterate loop):**
- 7B  `spider7b_iter50_cold_20260623_124313`  (GPU 1, accept bar `--min-accuracy 0.76` = beat IterGen-on-50 34/50 by ≥4)
- 1.5B `spider1p5b_iter50_cold_20260623_124313` (GPU 2, accept bar `--min-accuracy 0.66` = beat IterGen-on-50 29/50 by ≥4)
- COLD, Sonnet-4.6/Bedrock author, mask ON + bandit, **`--eval-max-steps 200` (= IterGen budget)**, 450s/example,
  **N=50 proportional subset of seed334 train** (easy12/med22/hard8/extra8; `itergen-on-50-bar.md`).
- Framework = the **2026-06-23 fairness-fixed build** (A1 scalar `3·acc+syn+0.25·delim+0.25·runtime_frac`,
  A2 counterfactual helper credit, A3 pareto seed in `feedback_loop.py`; B task-guidance removed from `prompts.py`).
- 3-stage discipline: iterate-on-50 → promote a bar-clearing candidate to 300-train → final one-shot 300-test
  (real win: 1.5B ≥159/300=53%, 7B ≥200/300=66.7%).
- _Predecessor_: the 300-set cycle-1 runs (`..._072721`/`..._073741`) died exit 120 after ~2 attempts each; the
  I1–I6 triggers below were logged from those 2 attempts and now carry over as live hypotheses for these runs.

Status legend: `OPEN` (waiting on evidence) · `VALIDATED` · `REFUTED` · `INCONCLUSIVE`.

---

## Ideas (append-only as they arise)

### I1 — Forced-constrained-from-token-0 is the key structural fix
**Logged:** 2026-06-23, after 1.5B att2 / 7B att2.
**Trigger:** Both authors, independently, diagnosed "wait for the model to emit `<<`" as failing and
pivoted to forcing `ConstrainedGeneration` from token 1.
**Hypothesis:** The winning (or best) strategy on *both* runs will be forced-constrained, not
wait-for-`<<`.
**Prediction:** The highest-accuracy attempt on each run uses forced/immediate constrained
generation; no high-accuracy attempt reverts to the wait-for-`<<` structure.
**Status:** OPEN.

### I2 — 7B syntax collapse is a verbosity problem, fixed by forcing constraint
**Logged:** 2026-06-23, after 7B att1 (syn 9.7%) vs 1.5B att1 (syn 98.7%).
**Trigger:** Near-identical att1 strategies, wildly different syntax — 7B emits prose/CoT instead of
bare SQL, and with no parser control `text_fallback` grabs non-SQL.
**Hypothesis:** Once the 7B forces constrained generation from token 0, syntax jumps to near the
1.5B's level (~95%+), because the parser guarantees SQL regardless of model verbosity.
**Prediction:** A 7B attempt that forces constraint from token 1 reaches syntax ≥ ~90%.
**Status:** OPEN. (7B att2 partial-pivot already moved syn 9.7%→25.0%.)

### I3 — The `!!!!!` dead-end wants DeadEndAvoidingStep / RollbackAndRegenerate
**Logged:** 2026-06-23, after 7B att2 rationale.
**Trigger:** Author observed the constrained decoder stuck repeatedly forcing `!` — a dead-end
state. The library already has `DeadEndAvoidingStep` + `RollbackAndRegenerate` (Defect-2 fix) for
exactly this, but the rationale isn't reaching for them.
**Hypothesis:** If a later attempt adopts a dead-end-avoiding step, the `!!!!!` pattern disappears
and syntax/accuracy improve; if no attempt ever adopts them, the dead-end recurs and caps quality.
**Prediction:** Either (a) an attempt that uses `DeadEndAvoidingStep`/`RollbackAndRegenerate` shows a
measurable jump, OR (b) the `!!!!!` pattern keeps appearing in rationales to the end. (Also tests
whether the bandit mask is even exposing these helpers to the author.)
**Status:** OPEN — leaning (b). 7B att2 rationale (verbatim) tackles the dead-end via
`AdaptiveConstrainedStep`'s "adaptive narrowness threshold," **not** `DeadEndAvoidingStep`/
`RollbackAndRegenerate`. Author hasn't reached for the dedicated helpers so far.

### I4 — Recurring Dafny postcondition verify failure burns iterations
**Logged:** 2026-06-23, after 7B att2 (`GeneratedCSD.dfy(89,2): postcondition could not be proved on
this return path`).
**Trigger:** Looks like the PROGRESS-postcondition-on-early-return issue the Pattern-A AuthorBody
restructure was meant to fix. The loop auto-recovers but each occurrence costs an iteration.
**Hypothesis:** If this verify error recurs across several iterations, the run spends its 20-attempt
budget fixing Dafny rather than searching strategy space → the AuthorBody fix doesn't cover this
return path and is worth revisiting.
**Prediction:** Count of distinct attempts that hit a verify failure. If ≥3–4 of 20, iterations are
being burned and this is real; if ≤1, it was a one-off and not worth chasing.
**Status:** **VALIDATED** (2026-06-23, mid-run). The monitor's verify-fail counter (greps Dafny
verify-fail lines in each run's log) reached **9 on the 1.5B run** by eval#12, and **5 on the 7B run**
earlier — far past the ≥3–4-of-20 "real" threshold on *both* runs. On the 1.5B run, ~9 of ~21 author
generations were burned re-fixing a Dafny error rather than searching strategy space (about 12 reached
eval). The loop auto-recovers each time (the error is sent back to the author, no crash), so it does NOT
block the run — but it roughly *halves the effective iteration budget*. **Takeaway:** the recurring
verify failure is real and costly; worth fixing the framework so iterations aren't spent on Dafny.
7B att2's rationale named the likely cause — `RegenerateUnitOnGroundingFailure` "returns a prefix of
unknown length … could exceed `|generatedPrefix| + maxSteps`", matching the recorded note that that
helper lacks a length `ensures` (memory: grounding-helper-spider1p5b) — i.e. the missing length bound,
not the early-return PROGRESS postcondition the Pattern-A AuthorBody fix targeted. **NOT yet verified:**
which exact postcondition fails on each of the 9/5 occurrences (the counts are from the log grep; the
per-failure root cause is inferred from the one 7B att2 rationale, not re-confirmed for each). Fix would
be a fair, mechanism-only change (add the length `ensures` to the unbounded helper / tighten its
contract) — candidate to land when the live runs are killed, alongside any I3/I6 helper-mapping work.

### I5 — The A1 scalar fix prevents a fast-but-worse attempt from being chosen as best
**Logged:** 2026-06-23, framing the whole point of this relaunch.
**Trigger:** Pre-fix, a low-accuracy no-timeout attempt outscored a high-accuracy one-timeout attempt
(att4 2.75 > att2 2.25). The new scalar weights accuracy 3×.
**Hypothesis:** The best-so-far / anchor selection now tracks the highest-accuracy attempt even if it
had a timeout, not a fast low-accuracy one.
**Prediction:** At run end, the recorded "best" attempt on each run is the highest-accuracy one (ties
broken sensibly), never a lower-accuracy attempt that merely avoided a timeout.
**Status:** OPEN.

### I6 — A clearer "situation → which helper" mapping might help the author pick the right tool
**Logged:** 2026-06-23 (user idea), after seeing the 7B author hand-roll a dead-end workaround
(`AdaptiveConstrainedStep` narrowness threshold) instead of using `DeadEndAvoidingStep`/
`RollbackAndRegenerate`, and cite an unknown-length helper that breaks verification.
**Trigger:** The author keeps reasoning its way *around* helpers that already solve the exact problem
it describes — suggesting the gap is "knowing which helper fits this situation," not capability.
**Hypothesis:** If the author had a clearer signal connecting a situation (dead-end, unbounded
length, grounding failure) to the helper that handles it, it would adopt the right helper sooner and
waste fewer attempts re-deriving workarounds.
**Admissibility (updated 2026-06-23 after the CLAUDE.md rule was corrected):** A "situation → which
helper" mapping is **ALLOWED**, as long as the situations are *decoding* situations (dead-end,
unbounded length, grounding failure) and not dataset-specific. Dead-end/length/grounding are pure CSD
mechanism — they'd mean the same thing on any dataset — so a mapping like "`DeadEndAvoidingStep`/
`RollbackAndRegenerate` exist for when the decoder gets stuck forcing one token" is fair to add to the
prompt or feedback. (Only a SQL/GSM/SMILES-specific mapping would be banned.)
**Three admissible forms, in roughly increasing strength:**
1. **Self-explanatory contracts** — give `RegenerateUnitOnGroundingFailure` the missing length
   `ensures` (see I4) and tighten postconditions so the author reads the guarantee off the signature.
2. **Richer eval feedback** — surface the observed failure shape ("decoder forced the same token N
   times → dead-end"; "span never entered") in the refinement signal, so the author infers the fix
   from data.
3. **Explicit mechanism mapping in the prompt** — a dataset-agnostic "when X decoding situation, helper
   Y handles it" list, naming the exact helper (e.g. "when the decoder is stuck forcing one token, use
   `DeadEndAvoidingStep`"). Fully fair: dataset-specific guidance and full-architecture-in-`--task` stay
   banned, but the discovery caveat (don't name the exact primitive) was retracted by the user
   2026-06-23, so naming helpers outright is allowed.
**Prediction / validation criterion:** Does the run's evidence show the author *failing to find* an
existing fit-for-purpose helper (I3/I4 turning out true)? If yes, that's the case for adding one or
more of forms 1–3.
**Status:** OPEN.

---

## Post-run validation pass

_(Filled when the runs finish or are cancelled. For each Ix: VALIDATED / REFUTED / INCONCLUSIVE,
with the specific evidence from the run — attempt numbers, acc/syn, rationale quotes — and a one-line
takeaway. Then a ranking of which ideas mattered most.)_

**Ran 2026-06-24.** Both 50-example COLD runs were killed ~20:01Z once each was clearly below bar.
Validated against the two runs that carried full attempt histories (NOT the `..._124313` predecessors,
which died early — 8 / 14 attempts). Both ran on the 2026-06-23 fairness-fixed build (A1–A4 + B done;
**A5 deferred**). Evidence = `grep` over the two run logs; line numbers are in that log.

- **7B** `spider7b_iter50_cold_20260623_173212` — 18 attempts, bar **76% acc / 85% syn**. Best = att6
  **30% / 88%**. BELOW (−46pp acc).
- **1.5B** `spider1p5b_iter50_cold_20260623_172808` — 14 attempts, bar **66% acc / 85% syn**. Best = att11
  **46% / 100%**. BELOW (−20pp acc).

**Per-attempt acc / syn (bar in header):**

7B (76/85): att1 12/24 · 2 0/0 · 3 4/0 · **4 46/64** · 5 0/2 · **6 30/88** · 7 0/0 · 8 28/84 · 9 30/82 ·
10 0/0 · 11 22/76 · 12 0/0 · 13 20/92 · 14 0/0 · 15 30/88 · 16 0/2 · 17 0/0 · 18 22/58

1.5B (66/85): att1 26/58 · 2 36/100 · 3 44/98 · 4 42/98 · 5 34/98 · 6 34/98 · 7 46/100 · 8 46/100 ·
9 42/100 · 10 38/100 · **11 46/100** · 12 44/98 · 13 26/98 · 14 40/100

**Anchor progression:** 7B att1→att4→att6 (then frozen att6 for 13 attempts). 1.5B att1→att3→att4→att11
(then frozen att11).

### Per-idea verdicts

- **I1 (forced-constrained-from-token-0 is the key fix) — SUPPORTED, not fully verified.** Both runs are
  `--no-require-delimiters` token-0; the best 7B attempt (att6) and the 1.5B family use span/forced
  constrained generation (`OpenConstrainedSpan + RegenerateUnitOnGroundingFailure + CloseSpanWithinBudget`
  per att6's rationale), and no high-acc attempt reverted to wait-for-`<<`. **Caveat:** I read this from
  attempt rationales, not a structural classification of all 32 attempts. Takeaway: consistent with I1, but
  forcing constraint alone did NOT get either run to bar (see I2).

- **I2 (forcing constraint fixes the 7B syntax collapse, syn ≥ ~90%) — VALIDATED (syntax only).** 7B syntax
  went from att1 **24%** to att13 **92%** (≥90% as predicted); several attempts cleared the 85% floor
  (att6 88, att8 84, att13 92, att15 88). **But high syntax ≠ high accuracy:** att13 was 92% syn / only 20%
  acc. Takeaway: the syntax-collapse problem is solved; the remaining gap is semantic, not syntactic.

- **I3 (the `!!!!` dead-end wants `DeadEndAvoidingStep` / `RollbackAndRegenerate`) — VALIDATED, branch (b).**
  `DeadEndAvoidingStep` = **0** occurrences and `RollbackAndRegenerate` = **0** occurrences in BOTH logs —
  the author NEVER reached for the dedicated dead-end helpers. The `!!!!` garbage pattern recurred to the
  last attempt (att18 rationale still discussing it). Instead the author kept inventing penalty/boost
  variants that aren't in its allowed list (see I6). Takeaway: the dead-end helpers exist and fit, but the
  author can't find/choose them — strongest single signal in the run.

- **I4 (recurring Dafny postcondition verify-fail burns iterations) — VALIDATED on 1.5B, REFUTED on 7B.**
  1.5B: **18** `postcondition could not be proved`, 24 `could not be proved`, 36 `GeneratedCSD.dfy(N,M):`
  error lines over 14 attempts → >1 verify-fix per attempt; roughly half the budget spent re-fixing Dafny.
  7B: **1** postcondition fail, 2 total Dafny error lines over 18 attempts → below the ≥3–4 threshold.
  **Reconciles the 5-vs-1 confusion:** the ledger's earlier "5 on the 7B run" was a *different/earlier* 7B
  run; on THIS completed run it's 1–2. Takeaway: Dafny verify-burn is a **1.5B-author** problem, not a 7B
  one — do not generalize the I4 fix priority to 7B.

- **I5 (the A1 scalar fix makes the recorded "best" track highest accuracy, never a fast low-acc attempt) —
  MIS-FRAMED; pareto anchor behaves correctly.** The prediction conflates two selectors. The *anchor* the
  author is told to beat uses pareto threshold-shortfall (A3), NOT the A1 scalar. On 7B the anchor froze at
  att6 (30%/88%) over the higher-accuracy att4 (46%/64%) — **by design**: att4 fails the 85% syntax floor
  (shortfall 0.30+0.21=0.51) while att6 clears it (shortfall 0.46+0=0.46), so att6 is correctly the
  threshold-closest. On 1.5B every contender clears syntax, so the anchor = highest-acc att11 (46%) as the
  prediction expects. Takeaway: the anchor is working as intended; the A1 *scalar* itself governs helper
  credit only and can't be validated from anchor behaviour (would need helper-mask logs). Prediction as
  written is REFUTED on 7B; the underlying mechanism is sound.

- **I6 (a situation→helper mapping / telling the author the real available set would help) — ORIGINAL
  VERDICT RETRACTED 2026-06-24 after a follow-up investigation.**
  - _Original (WRONG) claim:_ "VALIDATED — ≥4 confirmed helper-walls where the author called a helper not in
    its allowed list, producing ~0% syntax garbage; the deferred A5 gap is the top lever." I read the
    helper-name occurrences in the log as masked-helper violations without checking the mask state.
  - _What the follow-up found (verified):_ **Zero contract violations in BOTH runs** (`grep -c "contract
    violation\|Violations:"` = 0 for 7B and 1.5B). The adaptive mask never disabled a single helper the author
    then called — so the A5 sub-lever ("tell the author what's masked") had **nothing to act on** here. The
    helper-name lines I cited (L49297 etc.) were the author's **own rationale comments and call sites**, not
    violation messages.
  - _The helper is not buggy:_ `RolloutConstrainedWithPenalties` is Dafny-proven to return a grammar-valid SQL
    prefix (`ensures parser.IsValidPrefix(generatedOut)`, library L1979); it cannot emit `!!!!`.
  - _Real cause of the `!!!!` collapse — a strategy↔evaluator integration mismatch._ att5 output was
    `"SELECT COUNT(DISTINCT loser_name) \n!!!!!!!!"` (7B log L6932): valid partial SQL (the helper's clean
    output) + a degenerate `!` tail emitted **outside** the helper's grammar mask. The author opened a visible
    `<<` span; the Spider SQL extractor expects `SQL: <query>` text, not a `<<...>>` span (L13159–13160), and
    left the output in a shape the extractor mishandled ("0/50 visible `<<`", L6888). The author burned att2,
    att5, att6 rediscovering this interface mismatch.
  - _Corrected takeaway:_ the helper "gap" was **not** mask-invisibility and **not** a helper bug — it was the
    author fighting the visible-span ↔ SQL-extractor interface. Fixable without touching any helper. **A5 is
    NOT the binding lever.** This also intersects the open fairness flag about the `SQL:` extractor format
    matching IterGen's baseline (checked next).

### Ranking — which ideas mattered most

_(Ranking revised 2026-06-24 to match the corrected I6 above — A5/mask-visibility demoted.)_

1. **Author opens visible `<<` spans in token-0 (no-delimiter) mode (was mis-labelled "I6/A5 helper gap")** —
   the author burned att2/att5/att6 (and the analogous 1.5B attempts) fighting its own `OpenConstrainedSpan`
   habit: in token-0 mode the answer is grammar-constrained from token 0 with NO delimiters, but the strategies
   still emitted `<<`, leaving the output in a shape the extractor mishandles (valid partial SQL + degenerate
   `!` tail). Not a masked helper, not a helper bug, **and not a fairness problem** (fairness check below =
   clean). The lever: steer the author to pure token-0 grammar-constrained generation, no visible `<<`.
2. **The accuracy ceiling is semantic, not syntactic (from I2)** — 7B reaches 92% syntax but plateaus ~30%
   acc; the author's own diagnosis is `syntax_valid_semantic_mismatch` (right SQL shape, wrong tables/columns)
   on 29/50. No syntax fix moves this; needs a grounding/schema lever. **Likely the bigger blocker.**
3. **I4 asymmetry** — 1.5B wasted ~half its budget on Dafny verify-fixes (18 postcondition fails); 7B did
   not (1). A Dafny-contract fix helps the 1.5B author specifically.
4. **A1–A5 mask/scalar machinery was not the constraint** — the pareto anchor (I5) and reweighted scalar
   behave correctly, AND the mask never bound (0 violations), so A5 (mask visibility) is moot for these runs.
   The audit fixed real bugs but none of them were holding the bar.

**Bottom line:** the 2026-06-23 build fixed the right *bugs*, but none of them were the *binding constraints*.
Both runs are below bar for reasons orthogonal to the A1–A5 mask/scalar work: (7B) a semantic-accuracy ceiling
+ the visible-span↔extractor interface mismatch; (1.5B) Dafny verify-burn + the same interface mismatch. Next
change to test should target the span↔extractor interface and/or a semantic grounding lever — and any relaunch
is a billed COLD Bedrock+GPU run needing explicit OK.

### Fairness check — RESOLVED 2026-06-24: clean, no edge

The open `SQL:`/extractor fairness flag is closed. Verified on focal (the runnable repo; the local Mac repo is
stale and missing `delimited_output.py`):

- **Extractor shared.** Our CSD strategy and the IterGen baseline both extract via
  `extract_sql_scored_output` (`benchmarks/common/delimited_output.py`) and both score via the execution grader
  `prediction_matches_gold`. The IterGen baseline adapter (`run_legacy_fixed_strategy.py:902+`) generates with
  the real IterGen library (`from itergen.main import IterGen`) and then calls the **same** `logic.extract_actual`
  + `logic.is_correct` (L1024+).
- **Prompt identical under token-0 (the mode both seed334 runs used).** Driver launch line:
  `[launch] spider7b_iter50_cold_20260623_173212 … COLD, token-0, …`. `_token0_enabled()` defaults ON
  (`SPIDER_TOKEN0_CONSTRAINED != "0"`, `eval_logic.py:27`). In token-0 mode `format_prompt` returns
  `format_spider_itergen_aligned_prompt` (L95-96) = IterGen's EXACT bare `\nSQL:` prompt with the `quey` typo
  preserved byte-for-byte; no few-shot, no `<<>>` instruction. Both sides get it.
- **No `<<>>` edge.** Token-0 expects no delimiters; grammar `sql.lark` is shared and untouched. The author's
  strategies nonetheless emitted visible `<<` (`Contains << >>: yes (required: no)`, `visible_delimiters: no`)
  — a leftover habit from the pre-2026-06-22 legacy `<<>>` board. That *hurt* us (caused the att5 `!!!!`
  collapse), so it's a self-handicap, not an advantage.

**Real lever, final reframing:** steer the author off `OpenConstrainedSpan`/visible-`<<` patterns toward pure
token-0 grammar-constrained generation (the fair, aligned surface). "Constrain from token 0 rather than waiting
for a delimiter" is explicitly allowed CSD-mechanism guidance. The separate ~30% semantic-accuracy ceiling
(wrong tables/columns) is likely the bigger blocker and is independent of this.
