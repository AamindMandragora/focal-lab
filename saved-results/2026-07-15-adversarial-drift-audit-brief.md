# Adversarial drift audit brief

Date: 2026-07-15

## Purpose

Judge the current Claude helper-recovery deployment against the full campaign
contract, including decisions made before the provider migration. Do not treat
the provider switch itself as proof that the experiment queue is correct.

## Authoritative state

- Live code, processes, checkpoints, logs, and outputs:
  `/home/aadivyar/csd-generation` on `focal`.
- Local implementation worktree:
  `/Users/aadivyar/conductor/workspaces/Dynamic CSD Gen/marseille-claude-code-synthesis/csd-generation`.
- Results board:
  `/Users/aadivyar/conductor/workspaces/Dynamic CSD Gen/marseille-claude-code-synthesis/results_matrix.md`.
- Recovery manifest:
  `saved-results/2026-07-15-claude-helper-recovery-manifest.json`.

## Full campaign contract to judge

1. The `Unknown task` warm-recovery bug affected eight families: GSM
   Qwen3.5-2B/4B/9B, GSM Qwen2.5-14B, SMILES Qwen3.5-9B isocyanates, Spider
   Qwen3.5-4B/9B, and archived Spider Qwen2.5-7B. The user explicitly approved
   warm recovery for these contaminated families from the last clean evaluated
   strategy, with all earlier evaluated feedback reconstructed.
2. The general rule remains cold synthesis. The warm exception is limited to
   these contaminated recovery families; it must not silently become a new
   default.
3. Every higher-bar cycle must use the highest valid baseline accuracy target
   with the configured strict-win margin. The syntax target is the highest
   valid baseline syntax **clipped at 90%**. Archived rows are included.
4. The normal synthesis configuration remains: at most 40 attempts for these
   Qwen3.5 recovery cycles, adaptive helper masking on, bandit helper policy,
   refinement beam size 2, the original task text, model, split, sample count,
   max steps, token budget, and evaluator settings.
5. After one valid higher-bar cycle fails to find a winning CSD, discard that
   row from the results matrix and record why. Do not discard from a
   contaminated or incorrectly configured cycle.
6. A train acceptance is not a publishable win. Run the frozen held-out
   evaluation automatically; update `results_matrix.md` only from a completed
   held-out artifact. SMILES paper comparisons have historically used N=100
   held-out evaluation, and prior campaign notes explicitly warn against a
   50-train/100-held-out mismatch.
7. Do not change grammars, graders, scorer semantics, dataset splits,
   preprocessing, baseline prompts, or anything unavailable to the baseline
   methods. Helper changes must be general and no-gold.
8. Every author process must use a large reasoning author. For this deployment
   it must be `claude-sonnet-4-6` through the isolated first-party Claude Max
   account `aadivya@fermi.ai`, with no Bedrock or API-key fallback.
9. The final helper menu and warm-task fix must be loaded before new author
   calls. A process with an older imported prompt/helper surface is not enough.
10. Queue work greedily across all four GPUs, while keeping memory reservations
    safe and preventing duplicate workers. Orphaned vLLM engines must be
    cleaned up so they do not permanently hold memory.
11. Transient provider, SSH, process, and held-out failures must not turn a row
    into a scientific failure. Recovery should continue indefinitely, using
    hourly retries where a delay is needed.
12. One-time checkpoint claims must prevent accidental replay, but they must
    not make a legitimate transient crash unrecoverable.
13. The local combined log must include current and future queue logs plus
    actual sample text in real time. The phone supervisor should notify every
    completed synthesis attempt and held-out evaluation without stale-event or
    rapid-send floods.
14. Historical row-removal decisions remain authoritative unless the user
    explicitly reversed them or a proven contamination invalidated the cycle
    that supported the removal.

## Current live deployment observed before the adversarial review

- Active services: `csd-gsm14b-claude-helper-resume.service` and
  `csd-claude-recovery-queue.service`, both active with zero systemd restarts.
- Active rows: GSM14B attempt 55, GSM Qwen3.5-2B attempt 24, GSM Qwen3.5-4B
  attempt 25, Spider Qwen3.5-4B attempt 39, and Spider Qwen2.5-7B attempt 22.
- Queued rows: GSM Qwen3.5-9B and SMILES Qwen3.5-9B isocyanates.
- Excluded row: Spider Qwen3.5-9B because an earlier helper-recovery worker had
  already reached attempt 40.
- All active synthesis commands use Claude and account `aadivya@fermi.ai`; no
  active Bedrock synthesis command was found.
- Final recovery/provider/default test suite: 71 passed, 3 warnings. Wider
  suite: 117 passed. Helper/grounding suite: 20 passed.

## Concrete drift risks the judge must resolve

These are observations, not predetermined verdicts:

1. GSM Qwen3.5-2B/4B/9B manifest syntax thresholds are
   `0.918367346939`, but both the user ruling and `run_all_tests.py` define a
   `0.90` syntax ceiling. The live 2B/4B commands inherited the higher value.
2. The dedicated GSM14B command uses syntax threshold `0.85`, not `0.90`.
   Determine which pre-migration scientific target is authoritative for this
   recovery and whether this violates the later higher-bar policy.
3. SMILES Qwen3.5-9B isocyanates is configured for train N=50 and held-out N=50.
   Existing accepted SMILES rows and the prior alignment policy use held-out
   N=100; determine whether N=50 is an accidental recovery-manifest drift.
4. GSM14B is outside the six-row controller. Its dedicated launcher runs
   synthesis only; verify whether any path automatically launches held-out
   evaluation after a GSM14B win.
5. GSM14B's permanent one-time claim and `Restart=on-abnormal` may prevent
   recovery after a transient Claude/process failure. Verify the actual
   behavior rather than assuming the claim is sufficient durability.
6. Spider Qwen3.5-9B exhausted before the final helper-menu deployment and is
   excluded from the final queue. Determine whether the user's request to run
   all necessary cells after the helper fixes requires one more final-helper
   recovery for this row.
7. The combined local log is about 174 GB, while its current source logs are
   much smaller. The local data volume has only about 14 GB free and reports
   99% use. Determine whether the follower duplicated/replayed source history,
   whether continued operation risks filling disk, and how to preserve current
   logging without destructive data loss.
8. The phone supervisor log shows stale historical events and repeated warnings
   that send commands were invoked only five seconds apart. Verify that current
   Claude attempts and held-out completions are notified exactly once and that
   the monitor is not flooding or missing current rows.
9. The results matrix still contains internal GSM Qwen3.5-4B and 9B MetaDecode
   loss rows. Verify whether they are correctly pending the current valid
   recovery cycles or contradict earlier explicit removal decisions.
10. Verify source/hash parity for every deployed behavior file and service,
    and confirm no grammar, scorer, or split drift entered with the helper or
    provider changes.

## Required verdict format

The judge must first define what a complete, correct implementation requires,
then return `PASS` or `FAIL`. Every `FAIL` item must cite a live command, file,
artifact, or test and distinguish:

- a current correctness blocker;
- a historical/cost-only observation;
- an unresolved user decision that cannot be inferred;
- a harmless difference.

The implementation may be called complete only when a fresh judge returns
`PASS` with no unresolved current correctness blocker.
