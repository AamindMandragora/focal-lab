# AGENTS.md — `synthesis/evaluate/`

## Scope

**Evaluation orchestration**, runtime wiring, feedback loop hooks, and shared evaluation utilities—not benchmark-specific business logic.

## Rules

- Follow root **`AGENTS.md`**: benchmark-specific behavior belongs under **`benchmarks/<name>/`** (especially **`eval_logic.py`**).
- Preserve **DFA-mask / Syncode** integration assumptions documented in **`README.md`**; do not regress per-step validity to full-vocabulary parsing.
- **`evaluator.py`** should delegate; avoid growing monolithic if/else by benchmark.
- Keep CSD-authored prompt guidance capture generic: `AppendTaskGuidance`
  belongs in shared runtime/evaluation plumbing, not benchmark-specific scoring.
- **`run_legacy_fixed_strategy.main`** calls **`_ensure_repo_cache_env`** so subprocess CRANE runs inherit **`HF_HOME`**, **`HF_CACHE`**, **`TRANSFORMERS_CACHE`**, **`SYNCODE_CACHE`**, and **`ITER_SYNCODE_CACHE`** under the repository **`cache/`** unless **`CSD_CACHE_ROOT`** (or those variables) are already set; vendored **`syncode/syncode/common.py`** and legacy forks walk up to the same root when imports happen outside that entrypoint.
- **Prompt tiers:** baseline adapters render prompts via **`prompt_tiers.py`** and frozen assets under **`prompts/{benchmark}/`**. Tier 1 → `gcd` / `itergen` / `cars` (0 few-shot rows, templates end with `<<`); tier 2 → `unconstrained` / `crane` / `metadecode` (4 few-shot rows). CRANE **`main.py`** uses the same templates when **`legacy/CRANE/src/prompting/base.py`** resolves the repo root. Legacy GCD/IterGen/CARS use **`build_delimited_span_grammar`** on GSM, Spider, and SMILES Lark files (prompt supplies `<<`; grammar closes with `>>`); GCD uses `stop_words=[">>"]` on those datasets. See **`prompts/README.md`** and **`benchmarks/common/delimiter_grammar.py`**.
- Edits inside gitignored **`legacy/{CRANE,itergen,cars}`** require tracked patches under **`environment/legacy_patches/`** per **`environment/legacy/AGENTS.md`** (prefer fixing **`run_legacy_fixed_strategy.py`** when that suffices).

## See also

- **`README.md`** in this folder for component list and artifact paths.
- **`grammars/AGENTS.md`**, **`benchmarks/AGENTS.md`**, **`syncode/AGENTS.md`**.
