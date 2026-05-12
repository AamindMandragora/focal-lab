# `outputs/baselines/` — artifact notes

## Layout

- **Legacy fixed strategies** (`run_legacy_fixed_strategy`):  
  `baselines/<strategy>/<eval_model_slug>/<benchmark>__tb<step_token_budget>__ms<max_steps>.json`
- **Metadecode exports**:  
  `baselines/metadecode/<eval_model_slug>/<benchmark>__tb…__ms…__gen<generation_slug>__iter<synthesis_iters>.json`

Each file is minimal JSON: top-level `accuracy`, `syntax_rate`, and `answers[]` with `question` / `generated_answer`.

## Pin unless intentionally regenerated

Do **not** delete or blindly overwrite these **GSM-Symbolic** baselines unless you intend to re-run that experiment:

| Artifact | Role |
| -------- | ---- |
| `unconstrained/.../gsm_symbolic__tb1__ms900.json` | Unconstrained adapter reference row |
| `gcd/.../gsm_symbolic__tb1__ms900.json` | GCD / Syncode row (paired comparison) |
| `crane/.../gsm_symbolic__tb1__ms900.json` | CRANE adaptive row (paired comparison) |

Eval model slug below is **`Qwen_Qwen2.5_Coder_1.5B_Instruct`** (`Qwen/Qwen2.5-Coder-1.5B-Instruct`).

## Recorded metrics — GSM-Symbolic Phase‑1 defaults (`tb=1`, `ms=900`, ~100 samples)

Metrics below were taken from the JSON headers (`accuracy`, `syntax_rate`). Update this section whenever baselines are re-materialized.

| Strategy | Path (relative to `baselines/`) | Accuracy | Syntax rate |
| -------- | -------------------------------- | -------: | ----------: |
| unconstrained | `unconstrained/Qwen_Qwen2.5_Coder_1.5B_Instruct/gsm_symbolic__tb1__ms900.json` | 0.14 | 0.69 |
| gcd | `gcd/Qwen_Qwen2.5_Coder_1.5B_Instruct/gsm_symbolic__tb1__ms900.json` | 0.01 | 0.95 |
| crane | `crane/Qwen_Qwen2.5_Coder_1.5B_Instruct/gsm_symbolic__tb1__ms900.json` | 0.21 | 0.93 |
| cars | `cars/Qwen_Qwen2.5_Coder_1.5B_Instruct/gsm_symbolic__tb1__ms900.json` | 0.00 | 1.00 |

**CARS row caveat:** Accuracy **`0.00`** on this artifact reflected **scoring wiring**, not necessarily useless generations. GSM-Symbolic `extract_actual` only parses **`<< … >>`** spans (`synthesis/evaluate/benchmarks/gsm_symbolic/eval_logic.py`). Legacy CARS returned **bare expressions** (no delimiters), so **`actual`** stayed **`None`** and symbolic equivalence never ran — while grammar checks still reported **`syntax_rate = 1.0`**. The CARS adapter now wraps delimiter-free GSM bodies before scoring (`run_legacy_fixed_strategy._cars_normalize_gsm_symbolic_output`). **Re-run** that baseline JSON for a meaningful accuracy number.

### Pending / follow artifacts

- **IterGen** — expected path once the matrix completes the legacy adapter run:  
  `itergen/Qwen_Qwen2.5_Coder_1.5B_Instruct/gsm_symbolic__tb1__ms900.json`  
  (Lark `Tree.__deepcopy__` in legacy IterGen was patched for cyclic stacks so this job can finish.)
- **Metadecode + other models/benchmarks** — populate rows here after those JSONs land.

## Monitoring `run_all_tests.sh`

- Matrix logs are written under **`outputs/run_all_tests_<YYYYMMDD_HHMMSS>.log`** (stderr + stdout from the shell).
- While Python-heavy subprocess output used to appear stuck after model load, the harness sets **`PYTHONUNBUFFERED=1`** so progress tends to flush sooner—still **`tail -f`** your newest log while GPUs churn.
- Grep for **`Saved baseline JSON:`** / **`Wrote baseline JSON:`** to align disk artifacts with log checkpoints.
