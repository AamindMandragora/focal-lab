# Fast-iteration metaDecode-to-win — plan

**Goal:** every Qwen3.5 cell beats its competitor baseline (CRANE / IterGen / CARS)
on the **held-out / publishable** set — won by changing the metaDecode framework
itself, searched fast on a ~50-example set, then proven by a fresh COLD synthesis
on the full set.
**Source:** built from `planning/fast-iteration-metadecode-wins-questions.md`
(answered 2026-06-28).

## At a glance
- **Scope = 15 cells:** GSM ×3 + Spider ×3 + SMILES ×9 (3 classes × 3 sizes).
  Keep going until **all** win. (Spider-2B already wins; 14 to go.)
- **Order:** closest-to-winning first; start on the smaller models but run small
  AND big in parallel as GPUs allow.
- **Loop:** iterate COLD on the small set (fast search) → on a clear 50-set win,
  **fresh COLD synthesis on the full set** (no carry-over of the 50-set strategy —
  warm starts stay banned) → held-out re-eval is the number that counts.
- **Baselines rescored on the exact same small set** so the search bar is fair.
- **Levers are mine, failure-driven.** Hard constraint: never leak dataset-specific
  "how to win" guidance into any prompt / `--task` / feedback / helper description.
- **Budget: unbounded**, run overnight autonomously. Only $ = Bedrock author calls;
  all eval/baseline compute on focal is $0.

## The win pipeline (per cell)

    ┌─────────────────────┐
    │ 1. SMALL-SET SEARCH │  COLD synth, max-iter 40, eval every attempt on
    │   (fast iteration)  │  the ~50 set. Cheap + fast = many framework-change
    └──────────┬──────────┘  cycles per hour. This is a SEARCH SIGNAL, not the win.
               │  best attempt beats baseline-on-same-50 by ≥4 examples
               │  (SMILES: beat CARS UV by ≥0.04), at ≥ baseline syntax
               ▼
    ┌─────────────────────┐
    │ 2. FULL-SET PROMOTE │  FRESH COLD synth on the FULL train set (Spider 300;
    │  (prove it's real)  │  GSM 49; SMILES 50). NOT seeded from step 1. max-iter
    └──────────┬──────────┘  40. Confirms the framework (not one lucky strategy)
               │  crosses the full-train bar                can win at full scale.
               ▼
    ┌─────────────────────┐
    │ 3. HELD-OUT CONFIRM │  Pure re-eval ($0) of the accepted strategy on the
    │   (the real number) │  held-out set vs baseline-on-held-out. BOTH axes must
    └─────────────────────┘  beat. THIS lands in results_matrix as the win.

**Why step 2 is a fresh COLD run, not a promotion of the step-1 strategy:**
the 50-set strategy would be a warm start (banned), and a 50-set win can be luck
(exactly what bit Spider-9B: 73.7% train → 64.7% test). Re-synthesizing cold on
the full set tests whether the *framework change* reliably produces a winner, not
whether one strategy overfit 50 examples.

## How the pipeline specializes per dataset

    Dataset │ small set        │ full set      │ held-out         │ baseline
    ────────┼──────────────────┼───────────────┼──────────────────┼─────────
    Spider  │ train-50         │ train-300     │ test-300         │ IterGen
            │ (seed334)        │ (seed334)     │ (seed334)        │
    GSM     │ train-49         │ = small (49)  │ eval-49          │ CRANE
            │ (seed123 49×49)  │ no 300 exists │ (seed123 eval)   │
    SMILES  │ 50 samples/class │ = small (50)  │ fresh 50 sample  │ CARS
            │                  │               │ (diff seed)      │

- **Spider** is the only dataset with a real 50→300 shrink; that's where the
  speedup lives. Step 2 is a genuine 300-train COLD synthesis.
- **GSM** is already 49 examples. Step 1 = step 2 (the 49 IS the full set);
  step 3 = held-out re-eval on the eval-49 side. Must confirm the recorded CRANE
  bars (24.5 / 42.9 / 53.1) are scored on the **eval-49** side, else rescore.
- **SMILES** is generative (no fixed examples). "Held-out" = a fresh 50-molecule
  sample at a different seed, scored vs CARS on that same fresh sample.

## Cell order (closest-to-winning first)

    Wave  Cell          Current     Bar           Gap        Models
    ────  ────────────  ──────────  ────────────  ─────────  ──────
     —    Spider-2B     38.3% test  37.7%         WON ✓      (done)
     1    Spider-9B     64.7% test  67.0%         −2.3pp     big
     1    GSM-2B        16.3% trn   24.5%         −8.2pp     small
     2    GSM-4B        (running)   42.9%         ?          small
     2    Spider-4B     53.7% trn   66.0%         −12.3pp    small
     3    GSM-9B        pending     53.1%         ?          big
     3    SMILES ×9     pending     CARS per-cell ?          all

Smaller models iterate far faster (GSM/Spider 2B/4B vs 9B ≈ 90s/example), so they
give more framework-change cycles per hour — but per A2 we run small AND big
together whenever GPUs are free, not strictly small-then-big.

## Framework-change direction (failure-driven, mine to choose)

Two recurring, well-characterized **fair** levers from the Qwen2.5 history — the
starting hypotheses, not a fixed plan:

- **Spider gap is semantic** (valid SQL, wrong tables/joins). IterGen's edge is
  schema-aware inference-time backtracking (`backward('column')` when a name isn't
  in the schema). Fair because it mirrors the baseline's own mechanism. Lever:
  a **schema-grounding enforcement / mismatch-feedback** primitive at the right
  (symbol) granularity — the redesign already scoped in earlier sessions.
- **"Model won't emit `<<` cleanly"** → the constrained span never opens →
  syntax/accuracy collapse (hit on both GSM and Spider). Lever: **token-0 /
  forced span-entry** mechanics (the H7 "token-0 constrained" change gave Qwen2.5
  +12–34pp); likely needs re-tuning for Qwen3.5.

Each framework change follows TDD (red test from the requirement → implement →
green) and a per-change targeted experiment (smallest sample that gives signal),
per the project rules. Direction adapts to what each cell's failures show.

**Big structural changes first, sized to the gap.** When a cell is far from its
baseline, open with the *biggest core changes* to the framework — **introducing a
new CSD** (a new constrained-decoding primitive the author can call) or **changing
the iteration style itself** (the synthesis/feedback loop mechanics, span-close
discipline, when/how spans open). Do NOT open a large gap with small helper-shape
or run-setting tweaks; those are for closing the *last* small distance after a big
structural change has already moved the cell most of the way. A large gap usually
means the CSD is missing a whole operation or the loop is shaped wrong — match the
size of the change to the size of the gap, then refine. Still fair: a new CSD or
iteration-style change is allowed only as a mechanism change, never as leaked
dataset-specific win guidance.

## Fairness guardrails (non-negotiable)

    ✓ FAIR                                   ✗ BANNED
    ─────────────────────────────────────    ──────────────────────────────────
    framework/helper code changes            grammar edits (gsm.lark, sql_spider,
    span-entry / grounding mechanics          SMILES) — fixed to baseline
    feedback-loop / search mechanics          grader/scorer edits diverging from
    anything the author model derives          the baseline
      at runtime from fair inputs             dataset-specific win hints in any
    baselines rescored on identical            prompt / --task / feedback / helper
      indices                                  description (at ANY sophistication)
                                              warm starts (all synth COLD)
                                              adaptive-helper-mask OFF

## Build order

1. **Finish 50-set baselines.** IterGen Spider-2B done (25/50 = 50.0%/90.0%).
   4B running; then 9B. (GSM/SMILES already at small scale → recorded bars stand,
   but verify GSM CRANE side = eval-49.) — unblocks fair 50-set bars.
2. **Wave-1 small-set search (COLD, max-iter 40):** Spider-9B and GSM-2B first,
   with the first framework change in place. Relay every attempt (acc/syn vs bar,
   judge by CSD_RATIONALE). — unblocks promotion decisions.
3. **On a clear 50-set win → fresh COLD full-set synth → held-out re-eval → record.**
4. **Roll the winning framework change across the other cells**; re-search any cell
   it plausibly helps. Add waves 2–3 (GSM-4B, Spider-4B, GSM-9B, SMILES ×9).
5. **After all cells resolve:** reconcile `results_matrix.md`, then ablations +
   final doc (deferred until winning strategies exist).

## Open risks / watch-outs

- **Some cells may hit a genuine semantic ceiling** fair CSD can't close (Qwen2.5
  Spider-7B did, even after the full pipeline: 62% test vs 65.7% bar). If a cell
  resists, the honest output may be "characterized loss," and the lever is a
  *framework* change, not more iterations. Flag early rather than burn budget.
- **50-set selection effects:** the 50 can be easier or harder than the full set
  (Spider-2B's 50 are easier: 50.0% vs 37.7%). The ≥4-example margin guards luck;
  step 2's full-set COLD re-synth is the real check.
- **Promotion margin (≥4 examples / ≥0.04 UV) is my default** — not pinned by you.
  Say if you want it tighter/looser.
- **GPU contention:** colleagues share focal; co-locate only $0 evals on their
  cards, never billed synthesis. Big-model (9B) synth needs a near-empty GPU.
