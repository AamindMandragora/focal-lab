# Evaluation prompt tiers

Frozen few-shot demonstrations and tier templates for baseline / MetaDecode evaluation.

## Layout

| Path | Purpose |
|------|---------|
| `{benchmark}/tier1.txt` | Answer-only template; no delimiters (grammar constrains the expression). GSM includes `{CARS_INFO}` when strategy is `cars`. Default **0** few-shot rows. |
| `{benchmark}/tier2.txt` | Chain-of-thought template; instructs models to put formatted answers (math / SQL / SMILES) inside `<<` `>>` |
| `{benchmark}/shots.json` | Frozen few-shot pool (up to 8 GSM rows); adapters cap usage via `prompt_tiers.fewshot_count_for_tier` |

Benchmarks: `gsm_symbolic`, `spider`, `smiles`.

## Tier assignment

| Tier | Strategies |
|------|------------|
| 1 | `gcd`, `itergen`, `cars`, `rs` |
| 2 | `unconstrained`, `crane`, `metadecode` (CoT when the compiled CSD uses free LM steps) |

Compiled **metadecode** picks tier from the CSD body on every synthesis eval iteration:
tier **1** when the strategy only uses fully constrained helpers (e.g. `OpenConstrainedSpan`
+ `ConstrainedStep`), tier **2** when it calls free-LM helpers (`UnconstrainedStep`,
`UnconstrainedChunk`, `UnconstrainedGeneration`, `CraneGeneration`). On **SMILES**, tier-1
prompt text may still use a delimited `<<`/`>>` suffix while **decoder grammar stays
tier-2 delimited**. See `strategy_uses_reasoning_prompt` and `configure_eval_prompts` in
`prompt_tiers.py`.

Logic lives in `synthesis/evaluate/prompt_tiers.py`. Benchmark `eval_logic.format_prompt*` delegates there.

## Regenerating shots

```bash
python -m synthesis.evaluate.prompts.write_frozen_prompt_shots
```

| Benchmark | Shot source |
|-----------|-------------|
| `gsm_symbolic` | CRANE `legacy/CRANE/src/prompt_templates/gsm_symbolic.yaml` (`cot.gsm` + `std.gsm`, `<<`/`>>` delimiters) |
| `spider` | CRANE `spider.yaml` + live schema text from the Spider dataset |
| `smiles` | Live class files under `benchmarks/smiles/data/*.txt` (not `shots.json`; see below) |

SMILES prompts are rendered by `prompt_tiers.render_smiles_cars_prompt`: legacy CARS class
instruction and all `Molecule:` exemplars from the class `.txt` file, with the response line
updated to require `<<` / `>>` for harness scoring. Tier 2 adds a `Reasoning:` line before
`Molecule: <<`. `shots.json` remains for regeneration checks only.

Tier templates mirror the same legacy `task_specification` / `std_instruct` / `cot_instruct` prose (with `<<` `>>` instead of `[[START]]` `[[END]]`). Spider tier-2 few-shots and the target block end with `Reasoning:` so models emit brief CoT then `<<SELECT ...>>`.

## Few-shot caps

| Tier | Default rows used | Rationale |
|------|-------------------|-----------|
| 1 (`gcd`, `itergen`, `cars`, `rs`) | 0 frozen GSM/Spider rows | Constrained decoders: minimal prompt, no 8-shot CoT block |
| 2 (`unconstrained`, `crane`, `metadecode`) | 4 frozen GSM/Spider rows | CoT baselines; CRANE `main.py` passes `--num_shots 4` |

SMILES always includes the full exemplar list from each class `data/*.txt` file (legacy CARS).

Override GSM/Spider per call with `render_benchmark_prompt(..., max_fewshots=N)`.

## Decode caps

| Benchmark | `benchmark_max_new_tokens` |
|-----------|----------------------------|
| `gsm_symbolic` | 600 |
| `spider` | 512 |
| `smiles` | 256 (tier-2 CRANE / unconstrained; tier-1 body cap 96) |

Legacy adapters use `effective_max_new_tokens(dataset, --eval-max-steps)` so CLI budgets cannot exceed these caps.

## CRANE subprocess

With **`environment/legacy_patches/CRANE/010-vas-prompt-tiers-base.patch`** applied, **`legacy/CRANE/src/prompting/base.py`** loads the same tier templates for `gsm_symbolic`, `spider`, and `smiles` when `main.py` runs from a checkout that contains `synthesis/run_synthesis.py`.

Compare against the old YAML prompter:

```bash
python -m synthesis.evaluate.scripts.compare_crane_prompter_prompts
```
