# Known-fix exact-zero baseline supersessions

Date: 2026-08-05

## Purpose

Preserve five invalid source artifacts and their tested replacements in a
separate versioned root. None of these replacements is accepted baseline
evidence because each still produced blank or one repeated malformed output.

## Launch and completion

- Pool PID: 705607
- GPU: 3
- Started: 2026-08-04T23:19:15Z
- Finished: 2026-08-05T01:26:40Z
- Replacement root:
  outputs/baselines/exact-zero-repair-20260804-known-fixes-v3
- Checkout: e8cce50c7e0563ffa1ad43051de32ae0ea6eab08
- Pre-launch test: tests/test_itergen_transformers_compat.py — 16 passed.
- Pool result: five exit-0 artifacts; five quarantined system failures.
- Synthesis block remained present throughout.

## Results

1. spider-qwen35-2b-itergen
   - Source SHA: 22c2932d77c656a687e8f5490fc710b8e663d150cf3ddae7827295e798499cca
   - Replacement SHA: 9833a5a9fe7db201cf9e1edf8abff29ffbf6641bdb481a1d929bef813a052397
   - Result: 0 accuracy, 0 syntax, 300 rows, 0 nonblank, 0 distinct.
   - Review: every answer is a whitespace-only tab sequence.
2. spider-qwen35-4b-itergen
   - Source SHA: 1affdc1ea99b3eabf23890846be4dfb67114c73149d252a90672ae0e72a2ea91
   - Replacement SHA: 074045bde09925d3812af664cb872f63dc31237a8b686c04aeeaaa227ae65b95
   - Result: 0 accuracy, 0 syntax, 300 rows, 0 nonblank, 0 distinct.
   - Review: every answer is exactly 176 spaces.
3. smiles-acrylates-qwen25-7b-crane
   - Source SHA: e553157664508922bd37426d2a57eb403690bbe86665d67be6bebf317f1ec665
   - Replacement SHA: a2ab94a1c9fa9637c3f0494a64cc4abc5e80824d71882a99f4f0c022818ea5d7
   - Result: 0 accuracy, 0 syntax, 50 nonblank rows, 1 distinct output.
   - Review: all rows repeat the same malformed 408-character SMILES string.
4. smiles-isocyanates-qwen25-1p5b-crane
   - Source SHA: c32d357eb5ad0f29e989a1074c23276d75fc33bb79b1e22aaaccb9fed532cc7b
   - Replacement SHA: 105b9909de9e5e7acd3d5f4d328584cc0ef7b850412fbc4f86d96a6e1d0138ec
   - Result: 0 accuracy, 0 syntax, 50 nonblank rows, 1 distinct output.
   - Review: all rows repeat the same malformed 557-character SMILES string.
5. smiles-isocyanates-qwen25-7b-crane
   - Source SHA: fd18f7c89495a330ecc576f6218d2bd14b874a2a883179ee699e08a8a0afd9c7
   - Replacement SHA: a76dc6fefe2a67d5e9f55f71991c609b4cdd02df0688e1103e5dc263633dd938
   - Result: 0 accuracy, 0 syntax, 50 nonblank rows, 1 distinct output.
   - Review: all rows repeat the same malformed 503-character SMILES string.

Every source and replacement artifact remains preserved. All five replacements
are excluded from corrected evidence pending a further tested repair.
