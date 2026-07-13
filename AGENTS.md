# AGENTS.md

## Scope

These instructions apply to work in this repository.

## Project Goal

Synthesize Dafny CSD strategies that verify, compile, and evaluate successfully, with the objective of outperforming the CRANE baseline on the same evaluation setup.

## Critical Prompting Rule

Do not include strategy guidance in synthesis prompts or task descriptions.
Neutral API reference is tool/contract content, not strategy guidance.

Allowed prompt content:
- Task objective.
- Tool signatures, neutral API reference, and formal contracts
  (preconditions, postconditions, types, ranges, mechanics, cost/state effects,
  proof obligations).
- Neutral documentation for CSD-authored evaluator prompt guidance, including
  `AppendTaskGuidance` placement and first-call-wins semantics.
- Verified method-body examples as contract/format examples, not benchmark
  answers or task-specific strategy prescriptions.
- Empirical refinement context from the current synthesis run, including
  measured failures, search memory, helper usage, and evaluation history,
  without prescriptive strategy advice.

Disallowed prompt content:
- Recommendations about which tool to use.
- Preferred or forbidden strategy patterns.
- Baseline-comparison hints that imply structure.
- Benchmark-specific answer hints, dataset shortcuts, or evaluation leaks.
- Procedural "NOTE" hints about when or why to apply a tool.

## Hypothesis Ledger Comes Before the Experiment

Before running any experiment meant to validate a hypothesis, write the
hypothesis in the experiment ledger first. The ledger entry must include the
hypothesis number, the single variable or tweak, prior belief, and falsifiable
prediction before code changes, launches, or measurements begin.

For the active metaDecode/Qwen3.5 campaign, the ledger is
`docs/experiments/metadecode-fast-iteration-log.md`. Autonomous runs may waive
waiting for user confirmation, but they do not waive the ledger-first rule.
Monitoring an already-running job does not need a new ledger row unless a new
hypothesis or tweak is introduced.

## Key Paths

- Top-level project directories: `synthesis/`, `environment/`, `legacy/`, `dafny/`, `cache/`, `outputs/`, `logs/`, and `experiments/` (archived manual assets only — not imported by the pipeline).
- Root entry points: `run_all_tests.py` (matrix), `run_tmux.sh` (tmux helper).
- Default GSM-Symbolic split: `experiments/splits/gsm_symbolic_crane_proportional_49x49_seed123.json`; default Spider split: `environment/benchmark_splits/spider_dev_proportional.json`. Extra seed/oracle/probe manifests live under `experiments/splits/`.
- Legacy baseline codebases (CRANE / IterGen / CARS): clone with `bash environment/clone_legacy_csds.sh` into gitignored `legacy/*`; tracked pointer `legacy/README.md`; harness-vs-upstream notes `environment/legacy/DIFFERENCES.md`; **any edit under `legacy/{CRANE,itergen,cars}` must be captured as patches under `environment/legacy_patches/`** (see `environment/legacy/AGENTS.md`).
- Dafny binary: set `DAFNY_PATH` when needed; otherwise the runner uses repo-local `dafny/dafny` only if present, then falls back to `dafny` on `PATH` or `~/.dotnet/tools/dafny`.
- OpenAI API key: `synthesis/.env`
- Hugging Face checkpoints and SynCode mask/parser pickles: repository `cache/` (set `CSD_CACHE_ROOT` to relocate; legacy CRANE/IterGen/GCD paths resolve here when unset).

## Core Files

- `synthesis/verify/library/GeneratedCSD.dfy`
- `synthesis/verify/library/VerifiedAgentSynthesis.dfy` (member index: `synthesis/verify/library/README.md`)
- `synthesis/generate/generator.py`
- `synthesis/verify/verifier.py`
- `synthesis/verify/compiler.py`
- `synthesis/evaluate/feedback_loop.py`
- `synthesis/evaluate/evaluator.py`
- `synthesis/run_synthesis.py`
- `run_all_tests.py` (matrix launcher at repo root)
- `synthesis/scripts/reevaluate_compiled_csd.py` (post-synthesis eval for matrix Metadecode cells)

## Pipeline Run Modes

Use `python3 -m synthesis.run_synthesis` from the repo root. Set `CUDA_VISIBLE_DEVICES` only when intentionally selecting an allocation. By default, **generation** uses **OpenAI** (`OPENAI_API_KEY` and `OPENAI_GENERATION_MODEL` / `--generation-model`); **evaluation** defaults to local vLLM with **`Qwen/Qwen3.5-2B`** (first matrix eval model) unless you pass other flags. Matrix model ablations must use direct hosted APIs and must not route through Bedrock.

- Quick smoke run (fast sanity check, low sample count):
  `CUDA_VISIBLE_DEVICES=2,3 python3 -m synthesis.run_synthesis --task "Solve math word problems with constrained symbolic expressions." --dataset gsm_symbolic --min-accuracy 0.0 --min-syntax-rate 0.0 --max-iterations 1 --eval-sample-size 1 --eval-max-steps 256 --output-name smoke_gsm`
- Standard GSM-Symbolic synthesis run:
  `CUDA_VISIBLE_DEVICES=2,3 python3 -m synthesis.run_synthesis --task "Solve math word problems with constrained symbolic expressions." --dataset gsm_symbolic --min-accuracy 0.4 --min-syntax-rate 1.0 --max-iterations 5 --eval-sample-size 20 --eval-max-steps 900 --output-name gsm_main`
- Spider synthesis run (SQL), usually with higher per-step token budget:
  `CUDA_VISIBLE_DEVICES=2,3 python3 -m synthesis.run_synthesis --task "Generate executable SQL queries from natural language questions." --dataset spider --min-accuracy 0.6 --min-syntax-rate 0.95 --max-iterations 5 --eval-sample-size 20 --eval-max-steps 900 --eval-step-token-budget 8 --output-name spider_main`
- Spider run with explicit split manifest:
  `CUDA_VISIBLE_DEVICES=2,3 python3 -m synthesis.run_synthesis --task "Generate executable SQL queries from natural language questions." --dataset spider --spider-split-file <path/to/split.json> --spider-split-name train --min-accuracy 0.6 --min-syntax-rate 0.95 --max-iterations 5 --eval-sample-size 20 --output-name spider_split`
- SMILES synthesis run (all classes or class subset):
  `CUDA_VISIBLE_DEVICES=2,3 python3 -m synthesis.run_synthesis --task "Generate valid molecules in the requested class." --dataset smiles --smiles-classes acrylates,chain_extenders,isocyanates --smiles-samples-per-class 10 --min-accuracy 0.5 --min-syntax-rate 1.0 --max-iterations 5 --output-name smiles_main`
- Hosted generation defaults to **OpenAI** (`OPENAI_API_KEY`, model `gpt-5.4` or `OPENAI_GENERATION_MODEL`). **`gpt5.5`** in `run_all_tests.py` uses OpenAI with synthesis author reasoning effort `xhigh` by default (`CSD_OPENAI_REASONING_EFFORT` / `OPENAI_GENERATION_REASONING_EFFORT` override). **`opus4.7`** uses the **Anthropic** backend (`ANTHROPIC_API_KEY`, optional `ANTHROPIC_OPUS_MODEL`) with adaptive thinking, `xhigh` effort, and summarized thinking by default. **`gemini`** uses the direct **Gemini** API (`GEMINI_API_KEY`, optional `GEMINI_GENERATION_MODEL`, default `gemini-3-pro-preview`) with `CSD_GEMINI_THINKING_LEVEL=high` by default. Bedrock and Bedrock-backed **`gemini-pro`** profiles are rejected by the matrix runner.
- Full matrix runs cover `gsm, spider, smiles` by default (`run_all_tests.py --benchmarks`). Default eval models: `Qwen/Qwen3.5-2B`, `Qwen/Qwen3.5-4B`, `Qwen/Qwen3.5-9B`, and `meta-llama/Llama-3.1-8B-Instruct`. Default fixed strategies: `unconstrained, gcd, crane, itergen, rs, cars` (add `metadecode` to `--strategies` for synthesis). Run matrix jobs with `--reuse-baselines` when cached baseline JSONs are acceptable; matrix Metadecode launches default to `--accuracy-win-margin 0.0`, `--max-tokens 32768`, `--restart-after-stuck-iters 0`, `--helper-selection-policy bandit`, `--refinement-beam-size 2`, `--eval-max-seconds-per-example 90`, and `--eval-min-examples-before-threshold-stop 15` unless a specific ablation intentionally changes one of those knobs. Syntax remains a thresholded floor, not a paper win margin.
- Full repository test sweep:
  `python3 run_all_tests.py`
- `run_all_tests.py` uses the active Python environment by default and verifies RDKit before starting a real matrix run. To select another prefix, set `VAS_CONDA_ENV=/path/to/env`; `VAS_RDKIT_CONDA_ENV` remains as a legacy alias. The launcher prepends `CONDA_PREFIX/lib` to `LD_LIBRARY_PATH` so SciPy/transformers wheels resolve `libstdc++` correctly; Syncode needs **`mxeval`** with bundled **`data/`** — run **`bash environment/install_mxeval_into_env.sh`** once per env (see **`environment/README.md`**).

## Synthesis Runs Start Cold

Never warm-start a synthesis run. Do not use `--initial-strategy-file` to seed
synthesis from a prior strategy, including a strategy from an earlier attempt
in the same run. Cross-split warm starts are also prohibited because they can
leak information between training and evaluation splits.

`--initial-strategy-file` remains valid for pure re-evaluation of an already
recorded strategy when `--max-iterations 1` and the acceptance bars are zero.
If synthesis fails, improve the framework and relaunch cold rather than
continuing from the failed run's best strategy. Historical warm-start rows in
`results_matrix.md` must be flagged when relevant but must not be removed
without user approval.

## Verify the Strategy Author Model

Before diagnosing synthesis quality, verify the author model from
`--generation-model` and `--generation-backend`. Quality runs must use a
large reasoning model, such as `gpt-5.4` through the OpenAI backend or
`us.anthropic.claude-sonnet-4-6` through Bedrock with thinking enabled and
high effort. Do not use a local small model such as a 7B Qwen model to author
strategies for a quality run; small local authors are permitted only for
explicit smoke or infrastructure checks.

## Evaluation Expectations

- Always compare synthesized strategy performance against a CRANE baseline on the same model/split/sample settings before claiming success.
- Maintain high syntax/format validity while improving accuracy.
- Fixed-strategy GSM baselines should use the local CRANE GSM rows across `unconstrained`, `gcd`, `crane`, `itergen`, `rs`, and `cars` so comparisons are row-aligned.
- For fixed-strategy baseline JSONs, do not infer valid syntax from missing legacy metadata. Annotate rows with benchmark parser checks or treat missing syntax booleans as invalid.
- For CRANE-backed GSM rows that lack `variable_types`, infer numeric symbolic identifiers from `gold_answer` before syntax checking.
- Keep the GCD GSM-Symbolic adapter scoped to constrained expression bodies after `<<`; wrap those bodies for scoring, finalize the longest parseable expression prefix, and restrict identifiers to numeric placeholders from the evaluation sample.
- For instantiated GSM rows without symbolic numeric variables, use numeric-only syntax checks; do not let arbitrary identifiers satisfy GSM expression syntax.
- Keep benchmark-specific evaluation behavior in `synthesis/evaluate/benchmarks/*/eval_logic.py` and keep `synthesis/evaluate/evaluator.py` focused on orchestration/delegation.

## Performance Constraint

When touching parser validity logic, preserve DFA-mask-based validity checks (Syncode `DFAMaskStore`) for `ValidNextTokens` behavior. Do not introduce brute-force O(vocab) Lark parsing for per-step validity.

## Operational Defaults

- Prefer GPUs `2,3` for local runs unless intentionally using another allocation.
- Keep changes minimal and localized. Occasionally scan through the repo and cut fat, because bloat is the enemy of progress.
- Do not remove or alter formal contracts in Dafny files unless required by the task.
- Do not create, edit, delete, or commit files under `paper/` unless the user explicitly requests changes there (the paper tree is out of scope for routine agent work).
- When asked to modify files under `paper/`, do not run paper compilation checks; after making the requested edits, always automatically proceed with `git add`, `git commit`, and `git push` within the subdirectory to update Overleaf.
- When using adaptive helper masking or beam refinement, keep selection rules empirical/contract-based (measured metrics, verifier checks), not heuristic strategy advice in prompts.
- For bandit-style helper selection, keep exploration/exploitation policy in pipeline code/CLI knobs (e.g., UCB parameters), not in synthesis prompt prose.
- One-off experiment scripts, historical strategy `.dfy` bodies, and non-default split JSONs belong under **`experiments/`**, not the repository root.
- Always update the `README.md` and `AGENTS.md` local to the folder you made changes in, and the global `README.md` and `AGENTS.md` for large changes. Occasionally scan the repo at the end of a request to ensure they are up-to-date.
- Under `synthesis/`, update the nearest subdirectory **`README.md`** / **`AGENTS.md`** when behavior or conventions change (see `synthesis/README.md` for the layout); do not add documentation inside vendored `synthesis/evaluate/syncode/syncode/` except via the root `evaluate/syncode/AGENTS.md` policy unless upgrading the vendor drop.
- **Legacy upstream trees:** never leave manual edits only under gitignored **`legacy/CRANE`**, **`legacy/itergen`**, or **`legacy/cars`**. Add matching unified patches under **`environment/legacy_patches/<name>/`**, refresh **`environment/legacy/DIFFERENCES.md`** when behavior changes, and verify with **`bash environment/clone_legacy_csds.sh`** (see **`environment/legacy/AGENTS.md`**).
