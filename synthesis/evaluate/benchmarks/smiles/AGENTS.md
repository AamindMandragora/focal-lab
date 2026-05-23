# AGENTS.md — `synthesis/evaluate/benchmarks/smiles/`

## Scope

**SMILES** class-constrained molecular string benchmark: prompts, class-specific grammars, RDKit-backed checks.

## Rules

- **Metadecode / synthesis feedback** evaluates **all configured classes in one run** (`--smiles-classes` comma list; per-example grammars via **`build_dynamic_parser`**). **Fixed-strategy baselines** stay **one JSON per class** (`smiles__class_<name>`).
- Legacy single-class-only `get_grammar_file` restriction is removed; the returned path is bootstrap-only when multiple classes are active.
- Multi-sample runs must update prompts via **`prompt_state.py`** (good = novel valid in-class; bad = syntax-invalid, wrong-class, or exemplar-copy attempts; duplicates of a prior good/bad SMILES are not re-listed). Wire through **`init_prompt_states` / `apply_prompt_state` / `record_prompt_result`** in **`eval_logic.py`**. Legacy CRANE subprocess parity: **`environment/legacy_patches/CRANE/030-vas-smiles-prompt-state-grammar.patch`**.
- **Synthesis prompts:** `configure_smiles_eval_prompts` picks tier-1 (answer-only + `<<`/`>>` suffix) vs tier-2 (CoT) from the compiled CSD; **grammar stays tier-2 delimited** for metadecode. All tiers include an explicit **no verbatim exemplar reuse** task clause.
- **Syntax scoring:** tier-specific decoder grammars (especially tier-2 `start: smiles ">>"`) may reject extracted SMILES bodies. **`metrics.grammar_valid_with_fallback`** tries tier grammar (and tier + `>>` when delimited), then falls back to the class **base** body grammar (`base_grammar_text` on dataset rows). **`syntax_valid`** requires grammar + RDKit when RDKit is installed.
- Grammar text and assets under **`data/`** must stay in sync; update **`data/AGENTS.md`** when adding classes or files.

## See also

- **`README.md`** and **`data/README.md`**.
