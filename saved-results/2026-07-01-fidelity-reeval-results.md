# Fidelity Re-Eval Results - 2026-07-01

## Purpose

Record the empirical fidelity re-eval requested in the 2026-07-01 handoff:
compare real baseline adapters against hand-written metaDecode reconstructions on
the same local vLLM evaluator path.

No paid author calls were used. These were `--max-iterations 1` reconstruction
re-evals with `--initial-strategy-file`, which is pure re-evaluation rather than
synthesis.

## Inputs

- Focal repo: `/home/aadivyar/csd-generation`
- Python: `/apps/conda/aadivyar/envs/csd/bin/python`
- Eval model: `Qwen/Qwen3.5-2B`
- Backend: local vLLM on focal GPUs
- Result directory: `/tmp/fidelity_reeval_results`
- Master script: `/tmp/fidelity_reeval.sh`
- Master log: `/tmp/fidelity_reeval_master.log`

## Outputs

| Cell | Real baseline | Reconstruction | Evidence |
|---|---:|---:|---|
| CRANE/GSM, N=49 | 24.49% acc / 83.67% syntax from existing verified baseline JSON | 6.12% acc / 8.16% syntax | Existing baseline: `/home/aadivyar/crane_qwen35_baselines/thinkfix/crane_gsm_Qwen_Qwen3-5-2B_n49_s123_2b.json`; reconstruction: `/home/aadivyar/csd-generation/outputs/generated/fidelity_crane_gsm_20260701_101543_1f062a/results/success_report.json` |
| IterGen/Spider, N=100 | 5.00% acc / 52.00% syntax | 8.00% acc / 54.00% syntax | Baseline: `/tmp/fidelity_reeval_results/itergen_spider_legacy.json`; reconstruction: `/home/aadivyar/csd-generation/outputs/generated/fidelity_itergen_spider_20260701_111142_8c3897/results/success_report.json` |
| CARS/SMILES acrylates, N=50 | 28.00% UV / 100.00% validity | 8.00% UV / 96.00% validity | Baseline: `/tmp/fidelity_reeval_results/cars_smiles_legacy.json`; reconstruction: `/home/aadivyar/csd-generation/outputs/generated/fidelity_cars_smiles_20260701_112124_ea6990/results/success_report.json` |

## Algorithm

1. Run each real baseline through `synthesis.evaluate.run_legacy_fixed_strategy`.
2. Extract each reference strategy body into the `MyCSDStrategy` body format.
3. Run each reconstruction through `synthesis.run_synthesis` with
   `--initial-strategy-file` and `--max-iterations 1`.
4. Read JSON metrics from the baseline output JSONs and reconstruction
   `success_report.json` files.
5. Record the comparison here.

## Verified Details

CRANE/GSM:

- The attempted fresh CRANE legacy command failed before writing
  `/tmp/fidelity_reeval_results/crane_gsm_legacy.json`.
- Failure evidence in `/tmp/fidelity_reeval_results/crane_gsm_legacy.log`:
  `ValueError: The checkpoint you are trying to load has model type qwen3_5 but Transformers does not recognize this architecture`.
- The comparison therefore uses the existing verified Qwen3.5-2B CRANE baseline:
  `/home/aadivyar/crane_qwen35_baselines/thinkfix/crane_gsm_Qwen_Qwen3-5-2B_n49_s123_2b.json`.
- Existing baseline JSON: accuracy `0.24489795918367346`, syntax rate
  `0.8367346938775511`, N `49`, wall time `920.4615` seconds.
- Reconstruction report: accuracy `0.061224489795918366`, syntax rate
  `0.08163265306122448`, N `49`, correct `3`, denominator `49`, invalid excluded
  `0`, eval time `886.9475936889648` seconds.

IterGen/Spider:

- Baseline JSON: accuracy `0.05`, syntax rate `0.52`, N `100`, wall time
  `2448.3325` seconds, total generation time `2419.2287` seconds, mean generation
  time `24.192287` seconds/example.
- Reconstruction report: accuracy `0.08`, syntax rate `0.54`, N `100`, correct
  `8`, denominator `100`, invalid excluded `0`, eval time `467.99160528182983`
  seconds.

CARS/SMILES acrylates:

- Baseline JSON: accuracy/UV `0.28`, syntax/validity `1.0`, N `50`, wall time
  `94.7509` seconds, total generation time `84.4176` seconds, mean generation
  time `1.688353` seconds/example.
- Reconstruction report: accuracy/UV `0.08`, syntax/validity `0.96`, N `50`,
  correct `4`, denominator `48`, invalid excluded `2`, eval time
  `186.49727320671082` seconds.

## Recording Notes

- `results_matrix.md` was not changed in this pass.
- Reason: the CARS/SMILES re-eval baseline `0.28` UV / `1.0` validity conflicts
  with the existing Qwen3.5-2B acrylates CARS row `0.080` UV / `0.96` validity at
  `results_matrix.md:93`.
- The IterGen/Spider re-eval used N=100, while the active board's IterGen bar is
  N=300 at `results_matrix.md:75`.
- The fresh CRANE legacy command failed, so the CRANE comparison used an existing
  verified baseline JSON rather than a new JSON from this script.

## Follow-up CARS Helper-Fidelity Patch Check

After the first helper-fidelity patch on 2026-07-01, CARS reconstruction was
re-run from the isolated focal worktree:

- Worktree: `/home/aadivyar/.config/superpowers/worktrees/csd-generation/baseline-fidelity-helpers`
- Run directory: `/home/aadivyar/.config/superpowers/worktrees/csd-generation/baseline-fidelity-helpers/outputs/generated/fidelity_cars_smiles_penalty_patch_20260701_20260701_114157_ffd6dc`
- Report: `results/success_report.json`

Result:

| Cell | Real baseline target | Patched reconstruction | Evidence |
|---|---:|---:|---|
| CARS/SMILES acrylates, N=50 | 28.00% UV / 100.00% validity | 2.00% UV / 96.00% validity | Run output reported `Membership: 20.0%`, `RDKit Validity: 96.0%`, and metaDecode success report `Accuracy: 2.0%`, `Syntax: 96.0%` |

This patch did not make CARS reconstruction match the real CARS baseline.
