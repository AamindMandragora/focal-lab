# Exact-zero baseline repair and selective rerun

Date: 2026-08-04
Updated: 2026-08-05

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
        independent evidence and queue judge
                    |
                    v
 launch approved synthesis work in priority order
```

## Boundaries

- Work only in the existing focal worktree
  `/home/aadivyar/csd-generation-worktrees/full-baseline-campaign-20260803`
  on `codex/full-baseline-campaign-20260803`.
- Preserve every original baseline, log, status row, synthesis artifact, process,
  and cold-queue state file. New baseline outputs go under a versioned repair root.
- The original synthesis controller is stopped under the approved priority change.
  Do not launch synthesis while the repair monitor's block file exists.
- Use only GPUs `0,2,3`, and only when the queue releases enough memory. Preserve
  unrelated users' GPU processes. GPU `1` is out of scope without new approval.
- A repaired cell may still score zero. The target is a faithful, functioning
  baseline path, not a forced nonzero score. A diverse exact 0/0 requires a
  skeptical review bound to that artifact's SHA-256 before evidence can proceed.

## Verified starting evidence

- The campaign evidence contains 31 exact zero/zero baselines: two Spider
  Qwen3.5 IterGen cells and 29 SMILES cells across GCD, CRANE, and IterGen.
- Both Spider artifacts contain 300 identical whitespace-only completions.
- Each affected SMILES artifact contains 50 identical nonblank but malformed
  completions. In the current adapters, SMILES prompt history advances only
  after a syntax-valid molecule, so one deterministic invalid first answer
  leaves every later prompt unchanged.
- The original synthesis queue was active on GPUs `0,2,3`. Its selected cells
  were interrupted with completed and started attempts recorded; its controller
  is now stopped so the 31 baseline repairs have priority.

## Hypotheses and tests

| ID | Hypothesis | Current status | Evidence required before a fix |
|---|---|---|---|
| H1 | IterGen accepts ignored whitespace as complete before any Spider SQL symbol is produced. | Confirmed at output and parser-control level | A unit test reproducing accepted zero-progress whitespace and failing before the guard. |
| H2 | Qwen3.5 IterGen's manual cache/token update path makes whitespace the repeated top allowed token. | Likely contributor, exact trigger unknown | One-example token/progress trace comparing Qwen3.5 with working Qwen2.5 IterGen. |
| H3 | SMILES deterministic adapters repeat the same failure because prompt history changes only after a valid first output. | Confirmed by code and all 29 artifacts | A test showing an invalid first completion produces an unchanged second prompt. |
| H4 | Some SMILES decoders continue past an earlier complete molecule into a malformed repeated suffix instead of returning the completed prefix. | Likely | One-example parser timeline showing the first COMPLETE point and final emitted text. |
| H5 | Qwen3.5 CRANE prompt/chat formatting leaves thinking or prompt-echo text in the constrained completion. | Likely for the visible Qwen3.5 CRANE signatures | A one-example prompt/render trace and output-boundary assertion. |
| H6 | One or more zeros are faithful model failures after the adapter issues are removed. | Open | A clean smoke and full rerun with valid control flow that still scores zero. |

## Focused-test-verified repairs awaiting live reruns

- Commit `951ef778` hardens the synthesis block and evidence monitor and
  contains the tested SMILES GCD sampling repair. The first quarantined GCD
  cell still needs a versioned post-fix rerun.
- Focused tests identify bypassed Qwen3.5 chat formatting as the Spider IterGen
  blank-output cause. The repair renders only that model/dataset pair with
  thinking disabled while retaining the original scoring and evidence prompt.
  Both post-fix Spider baseline reruns are still pending.
- Focused tests show delimiter-free SMILES CRANE received the unreachable `>>`
  stop word. The repair removes it only for SMILES; the affected post-fix
  baseline rerun is still pending.
- Repeated malformed SMILES IterGen output remains quarantined. Sampling or
  changing prompt history would change strategy behavior and still requires
  user approval if no semantics-preserving harness fault is found.


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
  recomputation, GPU/process isolation, remaining author budgets, and queue order.
- Synthesis launches follow the approved order below and remain blocked until
  corrected evidence and the queue inputs are independently verified.

## Approved priority and recovery order

Approved on 2026-08-04 after the question round:

1. Interrupt GSM Qwen3.5-2B and SMILES chain-extenders Qwen2.5-1.5B now.
   Preserve completed attempts and count every started author call against each
   original 40-call cap.
2. Let SMILES acrylates Qwen3.5-4B finish attempt 40's training evaluation.
   Preserve any winning compiled CSD and defer held-out evaluation.
3. Run all 31 versioned baseline repairs before any synthesis restart. Use only
   GPUs `0,2,3`; GPU `1` remains out of scope.
4. Inspect the repair pool every five minutes. A new 0/0 is suspicious: other
   baselines may continue, but corrected evidence and synthesis stay blocked
   until structural output, parser progress, stop reasons, and logs pass review.
5. Bind every later job to one complete corrected evidence snapshot.
6. Run fresh 40-attempt cold campaigns for cells whose corrected target rises.
7. Retry the two earlier GPU-memory startup failures with a full-memory gate and
   separate output names.
8. Recover interrupted cells only when their target is unchanged. Restore
   completed history and use only the unused original author-call budget.
9. Run unchanged cells that never started, then deferred held-out evaluations.

The live monitor PID is recorded in
`.context/exact-zero-baseline-monitor.pid`; it polls every 300 seconds, writes
`saved-results/2026-08-04-exact-zero-baseline-monitor.json`, and holds
`.context/exact-zero-repair-synthesis.blocked` through all 31 row reviews,
corrected evidence creation, and independent queue validation.
If a monitor proves a harness fault, it may quarantine the artifact, add a
narrow test-first repair, and rerun only that cell. Strategy semantics and
prompt guidance still require user approval.

## 2026-08-05 authoritative repair update

This section supersedes the earlier progress counts and any earlier statement
that strategy changes still require approval.

### Current accepted denominator

- The first 31-cell repair pool is complete: 15 original artifacts are accepted
  and 16 are quarantined.
- One SHA-bound SMILES GCD supersession is independently accepted. Effective
  accepted evidence is therefore 16/31.
- Fifteen labels remain unresolved: two Spider IterGen, ten SMILES IterGen,
  and three SMILES CRANE.
- The five-cell known-fix pool completed, but all five replacements remain
  quarantined system failures. Preserve every source and replacement artifact.

### Approved repairs

1. Preserve Spider IterGen's recurrence feature. For repeated-token logits,
   multiply nonnegative logits by `0.3` and divide negative logits by `0.3`
   so both signs become less likely. Apply this only where the existing Spider
   recurrence penalty is active.
2. Use `do_sample=True` and `temperature=0.7` for SMILES IterGen and SMILES
   CRANE. Do not change GSM or Spider sampling behavior.
3. Apply neutral reasoning plus delimiters only to SMILES CRANE. Use the prompt:
   `Think through the requested molecular class, then put only the final SMILES between << and >>.`
   Reasoning is unconstrained before `<<`; grammar-constrained generation
   occurs inside the span; only the content inside `<< >>` is scored.
4. Do not add molecule examples, chemistry hints, preferred structures, or
   other strategy guidance.
5. Investigate Qwen3.5 chat rendering with first-token and prompt-boundary
   traces, but do not change it without a failing test and causal evidence.

### Proven Qwen3.5 prompt-cache defect

- A config-aware Qwen3.5 `DynamicCache` allocates empty layer objects before
  the first forward pass, so the cache is truthy while its sequence length is
  still zero.
- IterGen treated that truthy empty cache as populated and sent only the final
  prompt token on the first pass. The two Spider Qwen3.5 runs and the two
  acrylates Qwen3.5 runs that started before this repair remain quarantined.
- The repair checks `get_seq_length() > 0` before switching to one-token cached
  decoding. Tests cover both a truthy empty cache and a populated cache. The
  legacy edit is mirrored by patch `011-empty-config-cache-full-prompt.patch`.
- Accept no Qwen3.5 result from this repair until a live one-example probe or a
  post-repair pool worker demonstrates nonblank, prompt-specific parser
  progress. Rerun only the four pre-repair Qwen3.5 cells in another versioned
  root; preserve every v7 artifact.

### Test-first execution

1. Add red tests for sign-aware recurrence, SMILES sampling settings, CRANE's
   delimited prompt surface, and inner-span-only scoring.
2. Implement the smallest repairs, mirror any IterGen legacy edit in
   `environment/legacy_patches/`, and update the required nearby docs.
3. Run focused and adjacent tests, then search sibling uses of each changed
   generation setting and delimiter contract.
4. Run one-example smoke probes on approved free GPUs. Record prompt hashes,
   token progress, parser state, stop reason, and output diversity.
5. Rerun only the 15 unresolved labels into new versioned roots. Keep synthesis
   blocked and never overwrite prior artifacts.
6. Require skeptical validation for every result. Any diverse nonblank 0/0
   needs an independent SHA-bound review before acceptance.
7. Rebuild corrected all-five-baseline evidence and thresholds, run a separate
   final judge, then launch synthesis in the previously approved order.

## 2026-08-05 finalization update

### Accepted evidence

All 31 exact-zero labels now have accepted effective evidence:

- 15 accepted artifacts from the first repair root.
- 1 independently reviewed SMILES GCD sampling-v2 supersession.
- 11 accepted artifacts from the approved v7 root.
- 4 accepted Qwen3.5 cache-repair artifacts from the v8 root.

The final evidence builder must preserve those exact source and replacement
hashes and must reproduce the selection counts `15 + 1 + 11 + 4 = 31`.

### Corrected queue

```text
corrected all-five-baseline evidence
                |
                v
10 fresh changed-target cells (40 calls each)
                |
                v
2 full-memory retries (40 calls each, exclusive idle GPU)
                |
                v
2 interrupted GSM recoveries (36 + 39 remaining calls)
                |
                v
3 unchanged cells that never started (40 calls each)
                |
                v
3 held-out-only jobs (0 author calls, exact CSD SHA pins)
```

The total approved new-author-call cap is exactly `675`. Each phase must finish
before the next phase can dispatch. The corrected profile cannot exclude cells.

### Launch gate

1. Write the corrected evidence and queue manifest from the committed finalizer.
2. Validate all 31 evidence mappings, exact thresholds, recovery histories,
   held-out CSD pins, phase membership, and author-call accounting.
3. Obtain a separate `gpt-5.6-sol` review. Its approval file must bind the
   corrected evidence SHA-256, queue-manifest SHA-256, and pinned git commit.
4. Preserve the block record, remove the active block path only after approval,
   and launch on GPUs `0,2,3`. Recheck the block and approval before every job.
5. Verify the controller, child process, phase-one state, logs, and GPU ownership
   before declaring the synthesis queue running.

## Documentation and handoff

- Update the nearest `README.md` / `AGENTS.md` files for changed behavior.
- Save the final diagnosis, exact code/test commands, rerun manifest, hashes,
  result table, and threshold deltas under `saved-results/`.
- Commit only tracked repair code, tests, docs, and plan files. Do not stage the
  campaign's unrelated or generated artifacts.
