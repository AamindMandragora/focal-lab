# SMILES Bare-Output Rerun Queue

Date: 2026-07-08

Purpose: record the SMILES prompt-contract change from visible `<< >>` spans to bare
SMILES output, and queue the reruns needed to keep CARS and metaDecode comparisons clean.

## Change Being Queued

SMILES prompts now ask for one bare SMILES string after `Molecule:` and do not ask for
`<< >>` delimiters. SMILES CSD evaluation now treats parser-guided chunks as hidden chunks,
matching the existing task text that says the answer contract is a single SMILES string.

No grammar, scorer, dataset split, or SMILES class definition is changed.

## Why This Queue Exists

The CARS paper SMILES setup shows bare `Molecule:` examples and asks for a single SMILES
string. Our focal SMILES evaluator still had visible delimiter instructions in
`synthesis/evaluate/benchmarks/smiles/eval_logic.py`.

The stored CARS SMILES `generated_answer` strings already have zero visible delimiter rows,
so this queue is about prompt/evaluator cleanliness rather than fixing already-visible
output files.

## Executable Queue

Script:

```bash
.context/run_smiles_bare_prompt_queue.sh
```

The script writes new outputs under:

```text
outputs/controlled_comparison_bare_smiles/
```

It does not overwrite old `outputs/controlled_comparison/...` files.

## CARS Baselines Queued

The script queues every recorded SMILES CARS controlled-comparison JSON found in the current
focal board:

- Qwen2.5 1.5B and 7B, all three SMILES classes, N=100.
- Qwen3.5 2B/4B/9B short bars, all three SMILES classes, N=50.
- Qwen3.5 2B/4B/9B fresh bars, all three SMILES classes, N=100.
- The Qwen3.5 4B acrylates sanity run, N=3.

## Conditional metaDecode Queue

After the CARS reruns, the same script re-evaluates existing accepted Qwen3.5 metaDecode
SMILES CSDs under the bare-output evaluator:

- Qwen3.5-2B acrylates, chain_extenders, isocyanates.
- Qwen3.5-4B isocyanates.
- Qwen3.5-9B acrylates, isocyanates.

For each cell, it compares the new `accuracy` against the old held-out JSON. If the new
score is lower, it appends a COLD synthesis command to:

```text
.context/smiles_bare_prompt_paid_synthesis_todo.sh
```

That paid todo file is not launched by this queue. It must not be run without fresh billing
confirmation.

## Verification

Local focused code test:

```bash
python3 -m pytest csd-generation/tests/test_smiles_bare_output_contract.py -q
```

Result:

```text
3 passed in 0.06s
```

Focal prompt/evaluator checks:

```bash
/apps/conda/aadivyar/envs/csd/bin/python -m pytest \
  tests/test_smiles_bare_output_contract.py \
  tests/test_prompts_no_strategy_guidance.py \
  -q
```

Result:

```text
19 passed, 2 warnings in 3.95s
```

Queue script syntax check:

```bash
bash -n .context/run_smiles_bare_prompt_queue.sh
```

Result: exit code 0.

Final focal search:

```bash
rg -n "<<SMILES|SMILES.*<<|<<.*SMILES|>>.*SMILES|SMILES.*>>" \
  synthesis tests .context/run_smiles_bare_prompt_queue.sh \
  saved-results/2026-07-08-smiles-bare-output-rerun-queue.md
```

Result: remaining hits are the saved note, the negative assertion in
`tests/test_smiles_bare_output_contract.py`, docs that explicitly describe the new bare
contract, and old backup evaluator files named `eval_logic.py.bak_*`. No active SMILES
evaluator prompt still asks for `<<SMILES>>`.
