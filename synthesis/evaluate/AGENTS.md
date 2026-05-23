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
- **Prompt tiers:** baseline adapters render prompts via **`prompt_tiers.py`** and frozen assets under **`prompts/{benchmark}/`**. Tier 1 → `gcd` / `itergen` / `cars` (0 few-shot rows; **no** `<<` / `>>` in prompt text—grammar constrains the full answer). CARS GSM adds **`{CARS_INFO}`** in **`tier1.txt`** (symbolic-expression requirement). Tier 2 → `unconstrained` / `crane` / `metadecode` when the strategy uses free LM steps (4 few-shot rows; delimited spans). **SMILES synthesis** may use tier-1 prompt text with a delimited suffix while **`grammar_prompt_tier=2`** (`configure_smiles_eval_prompts`). SMILES prompts include an explicit **no verbatim reuse** task clause. Legacy tier-1 grammars use **`build_constrained_body_grammar`**; tier-2 / CSD paths keep **`build_delimited_span_grammar`**. See **`prompts/README.md`** and **`benchmarks/common/delimiter_grammar.py`**.
- **CARS baselines:** **`--cars-search-steps`** (default **200**); checkpoints **`output-json`** after each example. Artifacts: `…__cs<steps>.json`.
- **`unconstrained`** baselines use **greedy** decoding on GSM, Spider, and SMILES (`temperature=0` / `do_sample=False` in `run_unconstrained_*_adapter` and CRANE `main.py` defaults for GSM).
- Edits inside gitignored **`legacy/{CRANE,itergen,cars}`** require tracked patches under **`environment/legacy_patches/`** per **`environment/legacy/AGENTS.md`** (prefer fixing **`run_legacy_fixed_strategy.py`** when that suffices).

## See also

- **`README.md`** in this folder for component list and artifact paths.
- **`grammars/AGENTS.md`**, **`benchmarks/AGENTS.md`**, **`syncode/AGENTS.md`**.
