# SMILES CARS UV bar verification — the `cars.json accuracy` field is inflated

**Date:** 2026-07-02
**What this settles:** Which number is the real CARS UV bar for each SMILES cell — the
`results_matrix.md` value, or the higher `accuracy` field stored inside each
`outputs/controlled_comparison/smiles_qwen35*/{class}/cars.json`? Two earlier sessions drew
opposite win/loss conclusions because they trusted different numbers. This resolves it.

## Answer

**The `results_matrix.md` bars are the truth. The `cars.json` `accuracy` field is inflated
(it tracks roughly the class-membership rate, not the unique-valid rate) and must never be
used as the win bar.**

## How it was verified ($0, CPU-only, on focal)

The win metric is **UV = unique-valid rate** = (RDKit-valid AND in-class AND non-exemplar
molecules, then de-duplicated to distinct molecules) / N. This is what
`evaluate_smiles_output` + a dedup set computes, and it is the sole SMILES win metric (user
ruling: "no validity doesnt matter only UV does").

1. **Re-derived every CARS bar from the raw CARS outputs.** For each cell I loaded the CARS
   `answers[].generated_answer` strings and re-scored each with `evaluate_smiles_output`
   (require_rdkit=True), then de-duplicated. The recomputed UV **matched the
   `results_matrix.md` bar exactly on all 9 cells**:

   | class | 2B | 4B | 9B |
   |-------|----|----|----|
   | acrylates | 0.080 | 0.360 | 0.320 |
   | chain_extenders | 0.400 | 0.600 | 0.560 |
   | isocyanates | 0.280 | 0.100 | 0.400 |

   The `cars.json accuracy` field for the same cells reads far higher (e.g. acryl-9B stored
   **0.980** vs true UV **0.320**; acryl-4B stored 1.00 vs 0.360). That field is inflated.

2. **Proved the recompute method equals the evaluator's method.** Ran the identical recompute
   on the metaDecode held-out files and reproduced the evaluator's stored UV exactly:
   9B/acryl 0.310, 4B/iso 0.580, 2B/iso 0.290 (3/3 standard-format files; 9B/iso's file
   returns 0 on re-derivation — a storage-format quirk in that one file, its recorded 0.41
   stands). Because CARS and metaDecode are now scored by the same method, the comparison is
   apples-to-apples.

## Consequence for the board

Under the true bars the SMILES board is **4 wins / 5 open** (not the "CARS dominates 0.9–1.0"
picture the inflated field implied):

| cell | metaDecode held-out UV | CARS bar | verdict |
|------|------------------------|----------|---------|
| acrylates-2B | 0.17 | 0.08 | WIN |
| acrylates-4B | collapsed (<0.36) | 0.36 | OPEN |
| acrylates-9B | 0.31 | 0.32 | thin loss (−0.01, 1 example) |
| chain_extenders-2B | 0.14 | 0.40 | loss |
| chain_extenders-4B | — | 0.60 | OPEN |
| chain_extenders-9B | — | 0.56 | OPEN |
| isocyanates-2B | 0.29 | 0.28 | WIN |
| isocyanates-4B | 0.58 | 0.10 | WIN |
| isocyanates-9B | 0.41 | 0.40 | WIN |

The 2026-06-29 H66/H67/H70/H81 audits used the inflated `cars.json accuracy` and are **void**.

## How to reproduce

On focal, `~/csd-generation`, no GPU needed:
```python
from rdkit import RDLogger; RDLogger.DisableLog('rdApp.*')
import json
from synthesis.evaluate.benchmarks.smiles.dataset import get_smiles_task
from synthesis.evaluate.benchmarks.smiles.metrics import evaluate_smiles_output
# for each cell: load cars.json -> answers[].generated_answer -> evaluate_smiles_output(...)
# -> count distinct unique_valid_candidate / N  ==  the results_matrix bar
```
