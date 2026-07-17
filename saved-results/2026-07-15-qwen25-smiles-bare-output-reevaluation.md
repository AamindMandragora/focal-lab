# Qwen2.5 SMILES bare-output re-evaluation

Date: 2026-07-15

## Purpose

Replace the older visible-`<< >>` Qwen2.5 SMILES comparison with a direct
evaluation of the same six retained MetaDecode strategies and their six
matching CARS cells under the current bare-output evaluator.

This was evaluation only. It performed no synthesis, made no author-model
calls, and made no billed API calls.

## Fixed protocol

- Models: `Qwen/Qwen2.5-1.5B-Instruct` and
  `Qwen/Qwen2.5-7B-Instruct`.
- Classes: acrylates, chain extenders, and isocyanates.
- Methods: the retained cold-discovered, mask-on MetaDecode strategy and CARS.
- N: 100 generated answers per cell, 12 cells, 1,200 answers total.
- Current answer contract: bare SMILES after `Molecule:`. MetaDecode constrained
  generation begins inside a hidden chunk; CARS uses its bare-output adapter.
- Rolling prompt: enabled.
- MetaDecode constrained-token temperature: 0.7, matching the accepted older
  Qwen2.5 SMILES launchers.
- Maximum steps: 400. Token budget: 1.
- Models, stored strategy bodies, class definitions, grammars, scorer, rolling
  prompt behavior, N, step budget, and token budget were kept fixed.
- Output root:
  `/home/aadivyar/csd-generation/outputs/controlled_comparison_bare_smiles_qwen25_20260715/`.

The first full-start MetaDecode commands accidentally inherited the current
default temperature of 0.0. Process-environment inspection caught this before
they produced final N=100 JSON files. Those processes were stopped and excluded;
the retained MetaDecode runs explicitly used temperature 0.7.

## Independently recomputed current results

UV is unique-valid rate. Validity is RDKit validity. Diversity is mean pairwise
Tanimoto diversity.

| Model | Class | Meta UV | Meta validity | Meta diversity | CARS UV | CARS validity | CARS diversity | Meta-CARS UV gap |
|-------|-------|--------:|--------------:|---------------:|--------:|--------------:|---------------:|------------------:|
| Qwen2.5-1.5B-Instruct | Acrylates | 0.08 (8/100) | 0.80 | 0.732 | 0.04 (4/100) | 1.00 | 0.669 | +0.04 |
| Qwen2.5-1.5B-Instruct | Chain extenders | 0.44 (44/100) | 0.90 | 0.813 | 0.03 (3/100) | 1.00 | 0.863 | +0.41 |
| Qwen2.5-1.5B-Instruct | Isocyanates | 0.12 (12/100) | 1.00 | 0.588 | 0.06 (6/100) | 0.97 | 0.800 | +0.06 |
| Qwen2.5-7B-Instruct | Acrylates | 0.12 (12/100) | 0.98 | 0.611 | 0.03 (3/100) | 1.00 | 0.637 | +0.09 |
| Qwen2.5-7B-Instruct | Chain extenders | 0.73 (73/100) | 0.99 | 0.807 | 0.07 (7/100) | 1.00 | 0.820 | +0.66 |
| Qwen2.5-7B-Instruct | Isocyanates | 0.15 (15/100) | 0.98 | 0.673 | 0.04 (4/100) | 1.00 | 0.733 | +0.11 |

All six current pairs favor MetaDecode on the primary unique-valid metric. The
secondary axes are mixed: MetaDecode has lower validity in five of six pairs
and lower diversity in five of six pairs.

## Change from the older visible-delimiter measurements

| Model | Class | Older Meta UV | Current Meta UV | Change | Older CARS UV | Current CARS UV | Change |
|-------|-------|--------------:|----------------:|-------:|--------------:|----------------:|-------:|
| Qwen2.5-1.5B-Instruct | Acrylates | 0.10 | 0.08 | -0.02 | 0.07 | 0.04 | -0.03 |
| Qwen2.5-1.5B-Instruct | Chain extenders | 0.58 | 0.44 | -0.14 | 0.56 | 0.03 | -0.53 |
| Qwen2.5-1.5B-Instruct | Isocyanates | 0.23 | 0.12 | -0.11 | 0.20 | 0.06 | -0.14 |
| Qwen2.5-7B-Instruct | Acrylates | 0.14 | 0.12 | -0.02 | 0.08 | 0.03 | -0.05 |
| Qwen2.5-7B-Instruct | Chain extenders | 0.56 | 0.73 | +0.17 | 0.05 | 0.07 | +0.02 |
| Qwen2.5-7B-Instruct | Isocyanates | 0.19 | 0.15 | -0.04 | 0.17 | 0.04 | -0.13 |

H95 predicted that at least four MetaDecode cells would stay within 0.05 of the
older UV rate. Only three did, so the prediction was false. The current numbers
must replace the older values in paper-use comparisons.

## Verification evidence

- All 12 final JSON files exist and contain exactly 100 answers.
- Independent re-scoring loaded every raw `generated_answer`, then called the
  current `evaluate_smiles_output(..., require_rdkit=True)` and
  `smiles_trial_metrics` functions. It returned `errors: []`.
- The independent pass reproduced every stored MetaDecode accuracy, validity,
  diversity, membership, unique-valid count, and sample count.
- Evaluator source:
  `synthesis/evaluate/benchmarks/smiles/eval_logic.py`, SHA-256
  `80b0d1af04617a2b8e0dcec7db914c8137a31887ced2c7fc683e0a7eb90100e2`.
  Its file timestamp predates these runs and the verifier confirmed the bare
  prompt, hidden-chunk, and start-inside-constrained markers.
- Machine-readable audit:
  `/home/aadivyar/csd-generation/outputs/controlled_comparison_bare_smiles_qwen25_20260715/independent_recompute.json`.
- Final requirement-by-requirement audit:
  `/home/aadivyar/csd-generation/outputs/controlled_comparison_bare_smiles_qwen25_20260715/completion_audit.json`.
  It reports `status: PASS`, 12 final cells, 100 answers per cell, all six
  positive UV gaps, and zero independent recomputation errors.
- Focused evaluator checks passed on focal: `7 passed in 0.13s` for
  `tests/test_smiles_bare_output_contract.py` and
  `tests/test_smiles_rolling_suffix.py`.
- The final process check found zero active SMILES evaluation jobs.
- Three N=3 smoke files and
  `smiles_1p5B/isocyanates/cars.partial.json` (four answers) are non-final and
  explicitly excluded.

## Reuse

Use the six current rows above for Qwen2.5 SMILES paper tables and claims. Do not
mix them with the older visible-delimiter numbers. For an offline spot check,
run `rescore_smiles_unique_valid.py` on any final class directory and compare
its recomputed UV, validity, and diversity with `independent_recompute.json`.
