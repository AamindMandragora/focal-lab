# H70 acrylates-2B train analysis

**Date:** 2026-06-30

**Purpose:** capture the train-side signal from H70 while its held-out re-eval is still running, so the next SMILES hypothesis can be preregistered from the actual accepted strategy rather than from memory.

## Inputs

- Focal checkout: `/home/aadivyar/csd-generation`
- Train success report: `outputs/generated/smiles_qwen35_2b_acrylates_uv_qwen35_0627/smiles_qwen35_2b_acrylates_uv_qwen35_0627_20260629_215714_3a5627/results/success_report.json`
- Accepted compiled strategy: `outputs/generated/smiles_qwen35_2b_acrylates_uv_qwen35_0627/smiles_qwen35_2b_acrylates_uv_qwen35_0627_20260629_215714_3a5627/python/smiles_qwen35_2b_acrylates_uv_qwen35_0627_20260630_043509_609b52/GeneratedCSD.py`
- Held-out target still running at time of note: `outputs/controlled_comparison/smiles_qwen35_2b/acrylates/metadecode_uv.json`

## Observed train result

- H70 accepted on attempt **38/40**.
- Train accuracy / UV: **0.42**.
- Train syntax / RDKit validity: **0.78**.
- Training sample count: **50**.
- Evaluation `num_correct`: **39**.
- Evaluation `accuracy_denominator`: **39**.
- SMILES paper-trial auxiliary metrics:
  - `validity_rdkit`: **0.78**
  - `membership`: **1.0**
  - `unique_valid_count`: **21**
  - `sample_count`: **50**
  - `diversity_tanimoto`: **0.6121383341361687**
- Accepted strategy SHA256 from `strategy_code`: `73829d4f960ca0ae5ad633972258e147254fdb526da44a9be8c46b7fdf87c5b4`

## Algorithmic interpretation

1. The accepted H70 strategy does generate acrylate-class molecules: `membership=1.0`.
2. The live-bar train pass comes from enough unique valid acrylates, not from perfect validity: `unique_valid_count=21/50`, `validity_rdkit=0.78`.
3. The main remaining train-side bottleneck is diversity under validity, because the success report's own rationale says the best previous attempts traded validity against diversity and the accepted result still has fewer than half the samples as unique valid molecules.
4. Held-out must still be treated as unknown until `metadecode_uv.json` is rewritten by the live H70 held-out re-eval. Do not promote H70 until that artifact is current and clears the focal CARS UV bar **0.36**.

## Next-hypothesis implication

If H70's held-out result misses the **0.36** acrylates-2B live CARS UV bar, the next SMILES hypothesis should target **unique-valid diversity while preserving the acrylate core**, not class membership alone. A single-variable candidate is a diversity-biased post-core constrained decoding rule, compared against H70's accepted attempt-38 strategy, with the same model, class, bars, max iterations, and held-out path conventions.

