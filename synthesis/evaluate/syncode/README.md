# Vendored SynCode

Upstream [SynCode](https://github.com/uiuc-focal-lab/syncode) (grammar-guided LLM decoding with DFA mask stores), vendored under `syncode/syncode/`.

## What this repository uses

| Component | Used by |
|-----------|---------|
| `dfa_mask_store.py`, `parsers/`, `larkm/` (LALR path) | Synthesis + Metadecode evaluation via `benchmarks/common/parser_utils.py` |
| `infer.py`, `language_model.py`, `grammar_decoder.py`, `evaluation/*` | GCD/CRANE fixed baselines via `run_legacy_fixed_strategy.py` |
| `utils/sql_spider_eval/evaluation.py`, `process_sql.py` | Spider execution accuracy via `benchmarks/sql_spider/executor.py` |

## Policy

- Prefer fixes in first-party code under `synthesis/evaluate/` when possible.
- Patch vendored sources only when necessary; keep diffs minimal (see **`AGENTS.md`**).
- Nested `README.md` files under `syncode/syncode/` are mostly upstream documentation; this file is the project contract for the vendor drop.

## Cache

Set `CSD_CACHE_ROOT` or `SYNCODE_CACHE` / `ITER_SYNCODE_CACHE` so mask stores and parsers cache under the repo `cache/` directory (see root **`README.md`**).

## See also

- **`AGENTS.md`** in this folder.
- Upstream paper: [arXiv:2403.01632](https://arxiv.org/abs/2403.01632).
