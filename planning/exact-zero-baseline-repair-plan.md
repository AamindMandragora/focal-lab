# Exact-zero baseline repair and selective rerun

Date: 2026-08-04

## Shape

```text
hashed 100-cell evidence
          |
          v
31 exact 0-accuracy / 0-syntax cells
          |
          +--> Spider IterGen: whitespace-only decode (2)
          |
          `--> SMILES: repeated malformed decode (29)
                    |
                    v
      red tests + one-example reproductions
                    |
                    v
         smallest cause-specific repairs
                    |
                    v
       one-example smoke per failure signature
                    |
                    v
      rerun all 31 cells to a versioned output root
                    |
                    v
   recompute hashes, scores, maxima, and thresholds
                    |
                    v
 independent judge; stop before paid synthesis reruns
```

## Boundaries

- Work only in the existing focal worktree
  `/home/aadivyar/csd-generation-worktrees/full-baseline-campaign-20260803`
  on `codex/full-baseline-campaign-20260803`.
- Preserve every original baseline, log, status row, synthesis artifact, process,
  and cold-queue state file. New baseline outputs go under a versioned repair root.
- Do not stop, restart, or alter the running synthesis queue. Do not launch any
  paid author calls or replacement synthesis cells.
- Use only GPUs `0,2,3`, and only when the queue releases enough memory. Preserve
  unrelated users' GPU processes. GPU `1` is out of scope without new approval.
- A repaired cell may still score zero. The target is a faithful, functioning
  baseline path, not a forced nonzero score.

## Verified starting evidence

- The campaign evidence contains 31 exact zero/zero baselines: two Spider
  Qwen3.5 IterGen cells and 29 SMILES cells across GCD, CRANE, and IterGen.
- Both Spider artifacts contain 300 identical whitespace-only completions.
- Each affected SMILES artifact contains 50 identical nonblank but malformed
  completions. In the current adapters, SMILES prompt history advances only
  after a syntax-valid molecule, so one deterministic invalid first answer
  leaves every later prompt unchanged.
- The original synthesis queue is active on GPUs `0,2,3`; its artifacts and
  thresholds must remain untouched during this repair campaign.

## Hypotheses and tests

| ID | Hypothesis | Current status | Evidence required before a fix |
|---|---|---|---|
| H1 | IterGen accepts ignored whitespace as complete before any Spider SQL symbol is produced. | Confirmed at output and parser-control level | A unit test reproducing accepted zero-progress whitespace and failing before the guard. |
| H2 | Qwen3.5 IterGen's manual cache/token update path makes whitespace the repeated top allowed token. | Likely contributor, exact trigger unknown | One-example token/progress trace comparing Qwen3.5 with working Qwen2.5 IterGen. |
| H3 | SMILES deterministic adapters repeat the same failure because prompt history changes only after a valid first output. | Confirmed by code and all 29 artifacts | A test showing an invalid first completion produces an unchanged second prompt. |
| H4 | Some SMILES decoders continue past an earlier complete molecule into a malformed repeated suffix instead of returning the completed prefix. | Likely | One-example parser timeline showing the first COMPLETE point and final emitted text. |
| H5 | Qwen3.5 CRANE prompt/chat formatting leaves thinking or prompt-echo text in the constrained completion. | Likely for the visible Qwen3.5 CRANE signatures | A one-example prompt/render trace and output-boundary assertion. |
| H6 | One or more zeros are faithful model failures after the adapter issues are removed. | Open | A clean smoke and full rerun with valid control flow that still scores zero. |

## Test-first repair loop

1. Freeze the 31-label manifest with original artifact hashes and expected row
   counts. Add a content-quality gate that rejects whitespace-only batches and
   repeated malformed batches as completed evidence.
2. For each distinct signature, write the smallest test that fails on the
   current code. Run it and record the red result.
3. Add narrow diagnostic logs for prompt shape, parser progress, first complete
   prefix, stop reason, token count, and timeout. Never log prompts in bulk or
   any credentials.
4. Make the smallest repair that preserves each upstream strategy's intended
   semantics. Prefer the harness; if a legacy tree must change, add a tracked
   patch, refresh `environment/legacy/DIFFERENCES.md`, and verify a clean clone.
5. Re-run targeted tests and adjacent baseline tests. Search every sibling use
   of the changed parser, prompt, cache, and stopping contract.
6. Run one example per failure signature on a released approved GPU. Do not
   scale until the output shows real parser progress and the expected stop
   reason.
7. Rerun all 31 labels through a selective controller into a versioned repair
   root with separate logs, status, claims, and hashes. Never overwrite the
   original 31 artifacts.

## Completion gates

- Original 31 artifact hashes still match the frozen manifest.
- Every rerun has the exact expected row count: Spider `300`; SMILES `50`.
- No rerun is accepted merely because the JSON exists. The validator records
  nonblank count, unique outputs, malformed count, stop reasons, and scores.
- Every changed behavior has a red-before/green-after test and a live one-row
  smoke. Relevant existing tests pass.
- A new evidence file points to the repaired artifacts, recomputes exact counts,
  and derives thresholds from all five baselines without changing old evidence.
- A separate judge checks the repair semantics, hashes, row counts, score
  recomputation, GPU/process isolation, and the no-paid-rerun boundary.
- Work stops after reporting corrected baseline results and provisional threshold
  changes. Any replacement synthesis run requires a later explicit approval.

## Documentation and handoff

- Update the nearest `README.md` / `AGENTS.md` files for changed behavior.
- Save the final diagnosis, exact code/test commands, rerun manifest, hashes,
  result table, and threshold deltas under `saved-results/`.
- Commit only tracked repair code, tests, docs, and plan files. Do not stage the
  campaign's unrelated or generated artifacts.
