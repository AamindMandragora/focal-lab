# AGENTS.md — `synthesis/evaluate/`

## Scope

**Evaluation orchestration**, runtime wiring, feedback loop hooks, and shared evaluation utilities—not benchmark-specific business logic.

## Rules

- Follow root **`AGENTS.md`**: benchmark-specific behavior belongs under **`benchmarks/<name>/`** (especially **`eval_logic.py`**).
- Preserve **DFA-mask / Syncode** integration assumptions documented in **`README.md`**; do not regress per-step validity to full-vocabulary parsing.
- **`evaluator.py`** should delegate; avoid growing monolithic if/else by benchmark.
- **`run_legacy_fixed_strategy.main`** calls **`_ensure_repo_cache_env`** so subprocess CRANE runs inherit **`HF_HOME`**, **`HF_CACHE`**, **`TRANSFORMERS_CACHE`**, **`SYNCODE_CACHE`**, and **`ITER_SYNCODE_CACHE`** under the repository **`cache/`** unless **`CSD_CACHE_ROOT`** (or those variables) are already set; vendored **`syncode/syncode/common.py`** and legacy forks walk up to the same root when imports happen outside that entrypoint.

## See also

- **`README.md`** in this folder for component list and artifact paths.
- **`grammars/AGENTS.md`**, **`benchmarks/AGENTS.md`**, **`syncode/AGENTS.md`**.
