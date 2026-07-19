# Exhaustive train and held-out coverage prelaunch audit

Date: 2026-07-19

## Scope

The paper's main-result grid contains 27 MetaDecode cells:

- GSM-Symbolic: six models — Qwen2.5 1.5B/7B/14B and Qwen3.5 2B/4B/9B.
- Spider: the same six models.
- SMILES: fifteen cells — Qwen2.5 1.5B/7B and Qwen3.5 2B/4B/9B, each on acrylates, chain extenders, and isocyanates.

Separate author-model, token-budget, beam, and helper-mask ablations are not part of this main-result coverage goal.

## Cold synthesis inventory

Eleven cells need a new cold run for complete, fair main-result coverage.

| Dataset | Cell | Action | Evidence |
|---|---|---|---|
| GSM | Qwen2.5-1.5B | New cold run | Paper audit records the retained result as a warm start; warm synthesis is now banned. |
| GSM | Qwen2.5-7B | New cold run | Strict regrade changes the stored result to 46.9%; it also ran in the GSM completeness-bug period. |
| GSM | Qwen2.5-14B | New cold run | `results_matrix.md` still records the cell as planned; prior attempts did not yield a current fair row. |
| GSM | Qwen3.5-2B | New cold run | The paper audit deliberately excluded the old result; current fixed-code coverage is missing. |
| GSM | Qwen3.5-4B | New cold run | Old score was produced in the completeness-bug period and compared with an eval-side bar. |
| GSM | Qwen3.5-9B | New cold run | Old score was produced in the completeness-bug period and compared with an eval-side bar. |
| Spider | Qwen2.5-7B | New cold run | The 66.3% held-out win came from a warm recovery; a cold-only replacement is required. |
| Spider | Qwen3.5-4B | New cold run | Old recovery was warm and its train score was compared with a test-side bar. |
| Spider | Qwen3.5-9B | New cold run | User explicitly restored this previously excluded row on 2026-07-19. |
| SMILES | Qwen3.5-4B acrylates | New cold run | This is the only Qwen3.5 SMILES model/class cell with no reusable accepted strategy. |
| SMILES | Qwen3.5-9B isocyanates | New cold run | The old strategy's 2026-07-19 pure reevaluation completed at 0/100 UV and 100/100 syntax under the current scorer; matched CARS is 30/100. |

Sixteen cells have reusable cold strategies:

- Spider Qwen2.5-1.5B and Qwen3.5-2B: already audited cold train plus held-out rows.
- Spider Qwen2.5-14B: July-10 cold success at 70.67%/100% on train-300. The 2026-07-19 pure reevaluation on the current explicit test-300 split completed at 139/300 = 46.33% accuracy and 100% syntax (`outputs/reeval/exhaustive_0719/spider-qwen25-14b.json`); this current measured loss replaces the older stored 67.67% headline for the exhaustive table.
- All six Qwen2.5 SMILES cells: July-15 bare-output N=100 reevaluation, independently recomputed from all 1,200 generated strings with zero scoring errors.
- Qwen3.5 SMILES 2B/acrylates, 2B/isocyanates, and 4B/isocyanates: recorded cold strategies with held-out N=100 artifacts.
- Qwen3.5 SMILES July-10 `paid0708` cells: 2B/chain extenders, 4B/chain extenders, 9B/acrylates, and 9B/chain extenders. Their success reports record Sonnet-4.6, adaptive mask on, bandit selection, N=50 synthesis, and no initial-strategy field; each first author prompt is a fresh strategy request. Corrected held-out UV values are 0.510, 0.750, 0.460, and 0.660 respectively.

## Completed baseline measurements

- GSM CRANE train-49, in model order Qwen2.5 1.5B/7B/14B and Qwen3.5 2B/4B/9B: 8, 18, 20, 7, 16, and 15 correct. Each strict launch bar is exactly one additional correct example.
- Spider IterGen train-300, in the same model order: 160, 198, 201, 123, 195, and 201 correct. Final counts replay each raw completion against the canonical `train_indices` with this repo's official `execute_accuracy(..., etype="exec")` scorer because the IterGen rows' embedded gold/`exec` fields are misaligned. The normalized queue evidence records that rescore method and its fixed database/table paths.
- Matching SMILES queue baselines: Qwen3.5-4B/acrylates CARS = 18/50 and Qwen3.5-9B/isocyanates CARS = 18/50. Existing Qwen3.5-9B CARS N=100 raw-answer files also rescore to acrylates 12/100, chain extenders 34/100, and isocyanates 30/100 UV for the broader table audit.
- SMILES Qwen3.5-9B/isocyanates: the old compiled strategy's current N=100 pure reevaluation completed at 0/100 UV and 100/100 syntax (`outputs/reeval/exhaustive_0719/smiles-qwen35-9b-isocyanates-existing.json`), confirming that this cell needs a new cold run.
- Spider Qwen2.5-14B: current explicit-test pure reevaluation completed at 139/300 = 46.33% accuracy and 300/300 = 100% syntax; no author call was used.

## Queue behavior required for complete tables

Every new cell records its synthesis-time train result. An accepted strategy is evaluated once on the fixed held-out side without author credentials. If synthesis exhausts its cap, the driver selects the evaluated, compiled attempt with the smallest combined accuracy/syntax threshold shortfall and still runs that frozen strategy on held-out. The state is recorded as `complete_loss`, so a losing cell does not leave a blank held-out table entry.

## Paid launch gate

The user approved at most 480 Sonnet author calls, the broad $48-$720 estimate, AWS account `887730490125`, and `AWS_BEARER_TOKEN_BEDROCK` from focal `/home/aadivyar/csd-generation/.env`. Four interrupted initial-strategy calls are now counted against that ceiling. The relaunch manifest allows 476 more calls, so interrupted plus remaining calls still total 480. Scaling the prior eight-cell estimate gives about 73-87 GPU-hours without quota waits and 5-8 days with observed daily-token throttling.

## Queue driver verification

The implementation was built in focal worktree `/home/aadivyar/csd-generation-phaseb-python-fix` on branch `codex/phaseb-python-fix`. The 2026-07-19 main focused suite completed with 126 passing tests. A concrete prelaunch dry-run then exposed a broken direct-file systemd invocation; commit `46ca2a43b0ed9019eb2cc412f73130c86abd364f` changes it to `python -m scripts.runtime.run_cold_synthesis_queue`. The new test failed before that change and passed afterward; the post-fix focused subset completed with 81 passing tests, the real module printed its `--help`, and systemd unit verification exited successfully. The first service launch then exposed one unsupported queue-only CLI flag before any author call; commit `4e934595ffdb01e50746af5ed0449c2d5dc64c22` removes that duplicate flag and adds a command-to-entry-point compatibility test. The focused subset now completes with 82 passing tests.

The verified driver enforces the exact eleven-cell set, cold starts only, Sonnet 4.6 Bedrock author settings, canonical train-side synthesis, fixed held-out scoring without author credentials, held-out rows after accepted or exhausted synthesis, atomic result/state files, retryable blocked repairs, pinned code/data, exact comparator and train-split evidence, and restart validation of all saved provenance.

The concrete eleven-cell manifest, eleven normalized baseline records, combined evidence ledger, and train-baseline table are now materialized in the isolated manifest worktree. The driver loaded and validated all eleven records against their normalized and raw hashes, canonical splits, strict bars, cold-only commands, and pinned code commit. The older warm-recovery service and its restart monitor were found active during preparation and were disabled before the cold launch.

## First cold launch parser incident

The user approved AWS account `887730490125` for the recorded 480-call, $48-$720 range. At 2026-07-19 06:31 UTC the first four cells were dispatched, but each exited with code 2 because the queue passed the removed `--bar-split-name train` CLI option. The 06:31 segment in each log contains zero author banners, zero attempts, and zero Bedrock-call markers, so that first parser failure made no paid author request and produced no strategy state. Both services were stopped before repair. The compatibility test failed on the exact unsupported flag before the fix and passed afterward; the subsequent retry started each cell cold from attempt zero.

## Incident-monitor restart loop and call accounting

At 2026-07-19 06:44 UTC the repaired queue started, but the new incident monitor replayed the four already-handled 06:32 controller failures. Each failed repair incorrectly restarted the queue, producing five short startup cycles. The first four cycles ended before initial-strategy generation. The fifth cycle completed exactly one initial strategy for each of GSM Qwen2.5-1.5B, Qwen2.5-7B, Qwen2.5-14B, and Qwen3.5-2B, then the evaluations were interrupted when both services were stopped. The saved accounting file records each log's observed prefix byte count, prefix SHA-256, and marker counts: one `Generating initial strategy`, one `Strategy:`, and one `Attempt 1/...` per cell. These are four paid author calls. The interrupted strategies are not resumed or used as initial strategies; all relaunches remain cold.

Commit `bd03e2993981b3e737feed68f9aa574834481c80` changes the incident monitor so any invalid, rejected, failed, or exceptional repair leaves every recovery service stopped. It also stops services before independently rolling back deployed files and restoring the old attestation. The red tests reproduced the invalid-result restart, partial multi-service restart, post-start status exception, and live-verifier exception. Final evidence: five focused tests passed, the monitor file passed 31/31, the wider verifier passed 130/130, and an independent judge returned PASS.

Commit `f6ebfb20441cd2693506120e3055eb164afcf03b` accounts for those four calls in the queue contract. The four affected future caps are 39, 39, 79, and 39; the seven untouched caps remain 40. The future cap is therefore 476 and the total authorized exposure remains exactly 480. The campaign validator requires a nonnegative interrupted-call count for every cell and rejects any eleven-cell configuration whose remaining plus interrupted calls do not total 480. The budget test was red before the accounting fields existed; after the change the queue suite passed 20/20 and the wider verifier passed 130/130.

### GPU-isolated evaluator pin

The reviewed launch pin is now commit `3981f85d00a958f613b548052e5712c80cf49f8e`.
That commit makes the persistent evaluation pool respect the single numeric
physical GPU already assigned to each queue process through
`CUDA_VISIBLE_DEVICES`; its no-idle-slot fallback also stays on that assigned
GPU. Before the change, all four interrupted logs reported pool workers on GPUs
`[0, 1, 2]` despite distinct queue assignments, and the 14B run then failed with
only 375 MiB free on GPU 0. The focused tests were red 2/2 before implementation
and green 2/2 afterward. The no-model assignment probe returned
`{0: [0], 1: [1], 2: [2], 3: [3]}`, the wider runtime verifier passed 133/133,
and the independent judge returned PASS. The change does not alter a grammar,
grader, split, prompt, model, bar, worker-count policy, or scoring behavior.

### Isolated 14B startup repair and second interruption accounting

The 2026-07-19 07:54:59 UTC launch proved that the evaluator pools stayed on
their assigned GPUs, but Qwen2.5-14B then stopped at the shared synthesis vLLM
fraction `0.80`: its 16,384-token context required 3.0 GiB of KV cache and only
2.96 GiB was available. The incident monitor detected `gpu_memory_startup` at
07:56:44 UTC, stopped the cold queue, rejected an invalid automated repair, and
left the queue stopped. It did not automatically relaunch.

Commit `70b21f6cdf420614b1d23acfccaba2de294690e7` changes only the shared
synthesis vLLM fraction from `0.80` to `0.81`. The focused test was red 1/1 and
green 1/1; the wider verifier passed 134/134. A disposable one-example
Qwen2.5-14B probe on physical GPU 1 kept the 16,384-token context, loaded 27.57
GiB of weights, reported 3.35 GiB KV cache and 18,304 cache tokens, then exited
0 with one result. An independent judge returned PASS. The probe used local
evaluation only and made no Bedrock or other provider call.

The stopped launch completed one new initial author call in each of the same
four GSM logs. Each previous recorded prefix still matches its saved SHA-256,
and each appended portion has exactly one `Generating initial strategy`, one
`Strategy:`, and one `Attempt 1/` marker. Commit
`fc3b569c786ab474deaf5ed0bf1d825b1400f140` records two interrupted calls per
affected cell and caps their next cold runs at 38, 38, 78, and 38. The other
seven cells remain 40 with zero interruptions. The new launch exposure is 472
future calls plus 8 already interrupted calls, exactly the approved 480 total.
The manifest is pinned to `fc3b569c786ab474deaf5ed0bf1d825b1400f140`.
