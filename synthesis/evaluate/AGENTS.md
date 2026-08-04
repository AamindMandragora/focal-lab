# AGENTS.md — `synthesis/evaluate/`

## Scope

**Evaluation orchestration**, runtime wiring, feedback loop hooks, and shared evaluation utilities—not benchmark-specific business logic.

## Rules

- Follow root **`AGENTS.md`**: benchmark-specific behavior belongs under **`benchmarks/<name>/`** (especially **`eval_logic.py`**).
- Preserve **DFA-mask / Syncode** integration assumptions documented in **`README.md`**; do not regress per-step validity to full-vocabulary parsing.
- **`evaluator.py`** should delegate; avoid growing monolithic if/else by benchmark.
- Keep CSD-authored prompt guidance capture generic: `AppendTaskGuidance`
  belongs in shared runtime/evaluation plumbing, not benchmark-specific scoring.
- Attempt outcome ledgers must remain empirical: metrics, measured deltas,
  rationale-claim summaries, and observed failure-location counts. Include all
  small failure-location buckets rather than top-k truncating them.
- Aggregate failure-mode summaries in **`evaluator.py`** should list all
  detected mode buckets. Keep any cap on verbatim rollout examples separate
  from the aggregate counts.
- **`run_legacy_fixed_strategy.main`** calls **`_ensure_repo_cache_env`** so subprocess CRANE runs inherit **`HF_HOME`**, **`HF_CACHE`**, **`TRANSFORMERS_CACHE`**, **`SYNCODE_CACHE`**, and **`ITER_SYNCODE_CACHE`** under the repository **`cache/`** unless **`CSD_CACHE_ROOT`** (or those variables) are already set; vendored **`syncode/syncode/common.py`** and legacy forks walk up to the same root when imports happen outside that entrypoint.
- Edits inside gitignored **`legacy/{CRANE,itergen,cars}`** require tracked patches under **`environment/legacy_patches/`** per **`environment/legacy/AGENTS.md`** (prefer fixing **`run_legacy_fixed_strategy.py`** when that suffices).
- Keep the tracked IterGen compatibility adapter greedy-only: Transformers 5 may use an identity logits warper for `do_sample=False`, and models with linear-attention layers may replace IterGen's empty lazy cache with `DynamicCache(config=model.config)`, but do not approximate legacy sampling after `_get_logits_warper` removal.
- Keep CRANE result discovery scoped to the requested model, grammar mode, grammar, and chain-of-thought setting; never select the newest result across the whole dataset when baseline jobs run concurrently.
- Spider IterGen must use the checked-in upstream iterative column/table search,
  schema backtracking, 20-iteration limit, and greedy recurrence penalty 0.3;
  a single default-unit `forward()` is not the Spider protocol.
- Delimiter-free SMILES CRANE evaluation starts constrained at the first
  generated token. Do not wrap the grammar or prompt in `<< >>` markers.
- GCD SMILES evaluation samples at temperature 0.7 to avoid repeating one
  malformed output across the whole trial; GSM and Spider GCD evaluation stays
  greedy.
- GCD and IterGen SMILES adapters must honor the requested generation-token
  budget rather than silently capping a 400-token campaign at 256.

## See also

- **`README.md`** in this folder for component list and artifact paths.
- **`grammars/AGENTS.md`**, **`benchmarks/AGENTS.md`**, **`syncode/AGENTS.md`**.
