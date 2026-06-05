# AGENTS.md — `synthesis/evaluate/benchmarks/smiles/`

## Scope

**SMILES** class-constrained molecular string benchmark: prompts, class-specific grammars, RDKit-backed checks.

## Rules

- **Metadecode / synthesis feedback** evaluates **all configured classes in one run** (`--smiles-classes` comma list; per-example grammars via **`build_dynamic_parser`**). **Fixed-strategy baselines** stay **one JSON per class** (`smiles__class_<name>`).
- Legacy single-class-only `get_grammar_file` restriction is removed; the returned path is bootstrap-only when multiple classes are active.
- Multi-sample runs must update prompts via **`prompt_state.py`** (good = novel valid in-class; bad = syntax-invalid, wrong-class, exemplar-copy, or **duplicate** prior attempts; duplicates of a prior good SMILES are appended to **`bad_results`** so the next prompt discourages reuse). Wire through **`init_prompt_states` / `apply_prompt_state` / `record_prompt_result`** in **`eval_logic.py`**. Legacy CRANE subprocess parity: **`environment/legacy_patches/CRANE/030-vas-smiles-prompt-state-grammar.patch`**; CRANE baseline JSON export must call **`record_prompt_result` before `is_correct`** in **`_enrich_crane_baseline_rows`**.
- **Synthesis prompts:** `configure_eval_prompts` picks tier-1 (answer-only + `<<`/`>>` suffix) vs tier-2 (CoT) from the compiled CSD; **grammar stays tier-2 delimited** for metadecode. All tiers include an explicit **no verbatim exemplar reuse** task clause.
- **CARS baselines:** `run_cars_legacy_adapter` uses upstream `legacy/cars` oracle rejection (`learn_level=3`, `constrain_first=True` per [casa](https://github.com/large-loris-models/casa)), class `.lark` from `legacy/cars/datasets/smiles/` (not SynCode `constrained_body` wrapping), **`_legacy_smiles_benchmark_prompt`** for good/bad feedback, and the same `max_new_tokens` cap as rejection sampling.
- **Syntax scoring:** tier-specific decoder grammars (especially tier-2 `start: smiles ">>"`) may reject extracted SMILES bodies. **`metrics.grammar_valid_with_fallback`** tries tier grammar (and tier + `>>` when delimited), then falls back to the class **base** body grammar (`base_grammar_text` on dataset rows). **`syntax_valid`** requires grammar + RDKit when RDKit is installed.
- Grammar text and assets under **`data/`** must stay in sync; update **`data/AGENTS.md`** when adding classes or files.

## See also

- **`README.md`** and **`data/README.md`**.
