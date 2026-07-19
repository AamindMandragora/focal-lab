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

## Missing baseline measurements

- GSM CRANE train-49: all six model sizes. Qwen2.5-14B has completed; Qwen3.5 2B/4B/9B are running; Qwen2.5 1.5B/7B remain queued.
- Spider IterGen train-300: all six model sizes. Qwen3.5-2B = 123/300, Qwen3.5-4B = 194/300, Qwen3.5-9B = 200/300, and Qwen2.5-7B = 197/300 have completed. Qwen2.5-1.5B/14B remain queued.
- SMILES Qwen3.5-9B: no baseline run is needed. Existing CARS N=100 raw-answer files rescore to acrylates 12/100, chain extenders 34/100, and isocyanates 30/100 UV.
- SMILES Qwen3.5-9B/isocyanates: the old compiled strategy's current N=100 pure reevaluation completed at 0/100 UV and 100/100 syntax (`outputs/reeval/exhaustive_0719/smiles-qwen35-9b-isocyanates-existing.json`), confirming that this cell needs a new cold run.
- Spider Qwen2.5-14B: current explicit-test pure reevaluation completed at 139/300 = 46.33% accuracy and 300/300 = 100% syntax; no author call was used.

## Queue behavior required for complete tables

Every new cell records its synthesis-time train result. An accepted strategy is evaluated once on the fixed held-out side without author credentials. If synthesis exhausts its cap, the driver selects the evaluated, compiled attempt with the smallest combined accuracy/syntax threshold shortfall and still runs that frozen strategy on held-out. The state is recorded as `complete_loss`, so a losing cell does not leave a blank held-out table entry.

## Paid launch gate

Eleven cold launches allow at most 480 Sonnet author attempts. Scaling the prior eight-cell estimate gives about 73-87 GPU-hours without quota waits and 5-8 days with observed daily-token throttling. The recorded rough range of $0.10-$1.50 per author call gives a broad $48-$720 exposure, normally lower because accepted cells stop early. Billing source: `AWS_BEARER_TOKEN_BEDROCK` in focal `/home/aadivyar/csd-generation/.env`, work AWS account `887730490125`. Explicit approval of the expanded paid launch is required before Phase C.

## Queue driver verification

The implementation is isolated in focal worktree `/home/aadivyar/csd-generation-phaseb-python-fix` on branch `codex/phaseb-python-fix`. The 2026-07-19 focused suite completed with 126 passing tests. `git diff --check`, Python compilation, and systemd unit verification exited successfully; systemd printed one unrelated permission warning for `netplan-ovs-cleanup.service`.

The verified driver enforces the exact eleven-cell set, cold starts only, Sonnet 4.6 Bedrock author settings, canonical train-side synthesis, fixed held-out scoring without author credentials, held-out rows after accepted or exhausted synthesis, atomic result/state files, retryable blocked repairs, pinned code/data, exact comparator and train-split evidence, and restart validation of all saved provenance.

The concrete manifest is intentionally not created until the remaining no-cost baseline measurements finish and their normalized evidence files are built. No paid author call has been made during this preparation.
