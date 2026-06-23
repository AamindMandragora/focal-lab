# AGENTS.md — `synthesis/evaluate/benchmarks/smiles/`

## Scope

**SMILES** class-constrained molecular string benchmark: prompts, class-specific grammars, RDKit-backed checks.

## Rules

- **Metadecode / synthesis feedback** evaluates **all configured classes in one run** (`--smiles-classes` comma list; per-example grammars via **`build_dynamic_parser`**). **Fixed-strategy baselines** stay **one JSON per class** (`smiles__class_<name>`).
- Legacy single-class-only `get_grammar_file` restriction is removed; the returned path is bootstrap-only when multiple classes are active.
- Multi-sample runs must update prompts via **`prompt_state.py`** (good = novel valid in-class; bad = syntax-invalid, wrong-class, exemplar-copy, or **duplicate** prior attempts; duplicates of a prior good SMILES are appended to **`bad_results`** so the next prompt discourages reuse). Only **extracted** SMILES (`eval_row["smiles"]` / `extract_actual`) go into good/bad lists — never raw model output. When a duplicate attempt repeats an extracted SMILES already listed under bad results, append the capped full prior completion under a **Response:** block and an incrementing **`[repeat N]`** line so greedy runs cannot stall on an unchanged prompt. Wire through **`init_prompt_states` / `apply_prompt_state` / `record_prompt_result`** in **`eval_logic.py`**. Legacy CRANE subprocess parity: **`environment/legacy_patches/CRANE/030-vas-smiles-prompt-state-grammar.patch`**; CRANE baseline JSON export must call **`record_prompt_result` before `is_correct`** in **`_enrich_crane_baseline_rows`**.
- **Synthesis prompts:** `configure_eval_prompts` picks tier-1 (answer-only + `<<`/`>>` suffix) vs tier-2 (CoT) from the compiled CSD; **grammar stays tier-2 delimited** for metadecode. All tiers include an explicit **no verbatim exemplar reuse** task clause.
- **Pooled scoring:** `pooled_eval.py` defines `DEFAULT_SMILES_POOLED_SUCCESS_TARGET` (unique syntax-valid target per class), `DEFAULT_SMILES_POOLED_MAX_ATTEMPTS` (200), and `aggregate_smiles_pooled_scores`. Syntax/accuracy denominators use `success_target`, not attempt count. Greedy strategies use dynamic good/bad prompts; **`rs`** and **`cars`** use **static** prompts.
- **CARS baselines:** `run_cars_legacy_adapter` uses patched `legacy/cars` ([pparys/cars](https://github.com/pparys/cars)) oracle rejection (`learn_level=3`, `constrain_first=True`), class `.lark` from `legacy/cars/datasets/smiles/` (not SynCode `constrained_body` wrapping), and a **constant** pooled prompt (no good/bad suffix) so the oracle trie can learn across attempts.
- **Syntax scoring:** tier-specific decoder grammars (especially tier-2 `start: smiles ">>"`) may reject extracted SMILES bodies. **`metrics.grammar_valid_with_fallback`** tries tier grammar (and tier + `>>` when delimited), then falls back to the class **base** body grammar (`base_grammar_text` on dataset rows). **`syntax_valid`** requires grammar + RDKit when RDKit is installed.
- Grammar text and assets under **`data/`** must stay in sync; update **`data/AGENTS.md`** when adding classes or files.

## See also

- **`README.md`** and **`data/README.md`**.
