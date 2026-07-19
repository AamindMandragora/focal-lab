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

Eleven cold launches allow at most 480 Sonnet author attempts. Scaling the prior eight-cell estimate gives about 73-87 GPU-hours without quota waits and 5-8 days with observed daily-token throttling. The recorded rough range of $0.10-$1.50 per author call gives a broad $48-$720 exposure, normally lower because accepted cells stop early. Billing source: `AWS_BEARER_TOKEN_BEDROCK` in focal `/home/aadivyar/csd-generation/.env`, work AWS account `887730490125`. Explicit approval of the expanded paid launch is required before Phase C.

## Queue driver verification

The implementation was built in focal worktree `/home/aadivyar/csd-generation-phaseb-python-fix` on branch `codex/phaseb-python-fix`. The 2026-07-19 main focused suite completed with 126 passing tests. A concrete prelaunch dry-run then exposed a broken direct-file systemd invocation; commit `46ca2a43b0ed9019eb2cc412f73130c86abd364f` changes it to `python -m scripts.runtime.run_cold_synthesis_queue`. The new test failed before that change and passed afterward; the post-fix focused subset completed with 81 passing tests, the real module printed its `--help`, and systemd unit verification exited successfully.

The verified driver enforces the exact eleven-cell set, cold starts only, Sonnet 4.6 Bedrock author settings, canonical train-side synthesis, fixed held-out scoring without author credentials, held-out rows after accepted or exhausted synthesis, atomic result/state files, retryable blocked repairs, pinned code/data, exact comparator and train-split evidence, and restart validation of all saved provenance.

The concrete eleven-cell manifest, eleven normalized baseline records, combined evidence ledger, and train-baseline table are now materialized in the isolated manifest worktree. The driver loaded and validated all eleven records against their normalized and raw hashes, canonical splits, strict bars, cold-only commands, and pinned code commit. The older warm-recovery service and its restart monitor were found active during preparation and are now both disabled and inactive. No paid author call has been made during this preparation; the cold queue service remains uninstalled and stopped pending explicit billing approval.
