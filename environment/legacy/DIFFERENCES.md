# Legacy CSD upstream vs this repository

Upstream projects ship **standalone** research code. This repo **does not fork**
them in Git; instead it keeps optional local clones under **`legacy/`**
(gitignored except `legacy/README.md`) and drives them through
**`synthesis/evaluate/run_legacy_fixed_strategy.py`**.

The tables below separate **what upstream owns** from **what our harness adds**
so baseline numbers remain interpretable.

## Shared infrastructure

| Topic | Upstream expectation | Our harness |
|-------|---------------------|-------------|
| Hugging Face / Transformers caches | Often cwd-relative (`iter_cache/`, etc.) | **`_ensure_repo_cache_env`** sets `HF_HOME`, `HF_CACHE`, `TRANSFORMERS_CACHE`, and SynCode pickle roots under **`cache/`** via `CSD_CACHE_ROOT` unless already set (`run_legacy_fixed_strategy.py`). |
| SynCode Python package | CRANE / IterGen may vendor or install their own copy | CRANE subprocess runs with **`PYTHONPATH`** prefixed by this repo’s **`synthesis/evaluate/syncode`** + inner **`syncode`** package so baselines share the vendored DFA-mask stack with GCD. |
| GSM-Symbolic dataset JSON | CRANE ships `src/gsm_symbolic/` JSON fixtures | Evaluator defaults **`eval_runtime.gsm_source_dir`** to **`legacy/CRANE/src/gsm_symbolic`** when `CRANE_GSM_SYMBOLIC_DIR` is unset so GSM rows align across strategies. |

## CRANE (`legacy/CRANE`)

| Aspect | Raw CRANE | Our adapter |
|--------|-----------|-------------|
| Entrypoint | `python main.py` from `src/` with many CLI flags | **`run_crane_legacy_adapter`** invokes **`main.py`** with flags aligned to this repo’s benchmarks: `--dataset gsm_symbolic|spider|smiles`, delimiter symbols `<<` / `>>`, grammar modes `original` vs `adaptive`, optional SMILES class/sample overrides, model id from **`--eval-model`**. When **`--gsm-split-file`** / **`--spider-split-file`** are set (from **`run_all_tests`** manifests), **`main.py`** selects **`train_indices`** / **`test_indices`** before **`--num_examples`**, matching harness GCD/IterGen/CARS baselines. |
| Results consumption | Writes rich local artifacts | Loader **`_load_latest_crane_results`** + **`_annotate_legacy_rows_with_syntax`** augments rows with benchmark syntax validity and emits **minimal baseline JSON** (`accuracy`, `syntax_rate`, per-row answers). For SMILES, syntax uses shared **`evaluate_smiles_output`** (tier grammar with base-grammar fallback + RDKit) rather than raw tier-2 body-only parsing. |
| Prompting | `prompt_templates/*.yaml` + chat few-shot turns | **`prompting/base.py`** patch **`010-vas-prompt-tiers-base`**: when the repo root is visible, **`BasePrompter`** renders **`synthesis/evaluate/prompts/{benchmark}/tier{1,2}.txt`** (tier 2 when `--do_cot`, tier 1 otherwise) so CRANE/unconstrained subprocess runs share the same frozen shots as GCD/IterGen/CARS. Spider rows use plain `question` + `db_info` (**`020-vas-harness-utils`**). Verify with **`python -m synthesis.evaluate.scripts.compare_crane_prompter_prompts`**. |
| SMILES multi-sample prompts | Fixed prompt per class row | Patch **`030-vas-smiles-prompt-state-grammar`**: **`main.py`** applies **`SmilesPromptState`** before each prompt (good/bad molecule suffix via VAS tier-2 templates); **`utils.py`** stores **`base_grammar_text`** beside tier-2 **`grammar_text`**; **`prompting/smiles.py`** scores with base-grammar fallback and records prompt-state outcomes. Matches in-repo metadecode / GCD / CARS adapters. |
| GSM-Symbolic equivalence | Z3 with random substitution fallback on timeout/parse failure; two hard-coded gold expressions use substitution only | Patch **`040-vas-gsm-z3-only-equivalence`**: **`prompting/gsm_symbolic.py`** marks **`correct`** only when Z3 returns **`unsat`**; parse failures, solver **`unknown`**, and missing Z3 return false. Harness **`gsm_symbolic_z3_equivalence`** follows the same rule (no substitution). |

## IterGen (`legacy/itergen`)

| Aspect | Raw IterGen | Our adapter |
|--------|-------------|-------------|
| Import layout | Expects package **`itergen`** on `PYTHONPATH` | **`_itergen_add_import_paths`** prepends the repo root and nested `syncode` paths used by upstream layouts. |
| HF / SynCode cache | cwd-relative `cache/` | Patch **`010-vas-cache-and-hf-env`**: resolve **`CSD_CACHE_ROOT`** / repo **`cache/`**, normalize **`SYNCODE_CACHE`**, bfloat16 on CUDA, pad token fallback. |
| Transformers sampling | Uses **`model._get_logits_warper`** directly | Patches **`020-vas-language-model-compat`**, **`040-vas-main-harness`**: warper/processor split compatible with Transformers 4.4x+, empty **`stop_strings`** guard, safer beam-mode defaults. |
| Lark deepcopy | **`Tree.__deepcopy__`** can recurse on cyclic stacks | Patch **`030-vas-lark-tree-deepcopy`**: bounded copy for long GSM runs. |
| GSM grammar | Static Lark / SynCode bridge | We rebuild **`gsm.lark`** per evaluation batch via **`_legacy_gsm_symbolic_grammar_base`** (allowed symbolic identifiers from `variable_types`, else numeric-only), then apply the same **`syncode: start ">>"`** body tweak as GCD so decoding starts after the prompt’s opening `<<`. |
| Output normalization | Raw completion text | GSM completions pass **`_gsm_symbolic_completion_to_delimited`** so **`<<expr>>`** extraction matches **`benchmarks/gsm_symbolic/eval_logic.py`**. |

Tracked patches: **`environment/legacy_patches/itergen/`** (applied by **`clone_legacy_csds.sh`**).

## CARS (`legacy/cars`)

Upstream: **[pparys/cars](https://github.com/pparys/cars)** (paper reference implementation). The newer **[large-loris-models/casa](https://github.com/large-loris-models/casa)** package is a separate refactor and is **not** what our harness imports.

| Aspect | Raw CARS | Our adapter |
|--------|----------|-------------|
| API surface | `cars.lib.ConstrainedModel` experiments | **`run_cars_legacy_adapter`** constructs **`ConstrainedModel`**, injects per-example Lark grammar (GSM dynamic/numeric, Spider `sql.lark`, SMILES class grammar text), runs **`generate`-style loop**, scores through **`Evaluator` + benchmark `eval_logic`**. |
| Model loading | Requires **`secrets.json`**, fixed HF model list | Patch **`010-vas-lib-hf-cache-device-map`**: optional secrets, repo **`cache/`** via HF env vars, Qwen2.5-Coder 1.5B/14B ids, chat-template fallback for instruct models. |
| MCMC path | Same secrets assumption | Patch **`020-vas-mcmc-lib-hf-cache`**: shared HF cache dir resolution. |
| CLI entry | **`run_task.py`** model index table | Patch **`030-vas-run-task-harness`**: accepts Hugging Face model ids directly (matrix **`--eval-model`** strings). |
| GSM answers | May emit bare expressions | **`_cars_normalize_gsm_symbolic_output`** wraps delimiter-free bodies so **`extract_actual`** sees `<<…>>` spans (see `outputs/README.md` for baseline layout). |
| Spider syntax flag | N/A | Adapter sets syntax True when extracted SQL mentions **`SELECT`** (legacy rows lacked rich syntax metadata). |

Tracked patches: **`environment/legacy_patches/cars/`** (applied by **`clone_legacy_csds.sh`**).

## GCD baseline (no `legacy/` tree)

**`gcd`** uses **vendored SynCode** only (`synthesis/evaluate/syncode/`). Differences vs upstream SynCode releases are governed by the vendor-drop policy under **`synthesis/evaluate/syncode/`** docs—not by `legacy/`.

## Verified reference strategies (Dafny)

**`synthesis/verify/reference/*.dfy`** mirror baseline *ideas* for verification and paper exposition only. They are **not** line-level ports of `legacy/*` Python; see **`synthesis/verify/reference/README.md`**.

## Refreshing file-level diffs

Use **`python synthesis/scripts/report_legacy_upstream_diff.py`** (see `--help`). It compares your **`legacy/<name>`** trees against pristine clones (optional **`--fetch-upstream`**) so local edits or forgotten patches show up as unified diffs.
