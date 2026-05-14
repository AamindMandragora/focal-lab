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

## Key Paths

- Non-hidden project directories are intentionally limited to `synthesis/`, `environment/`, `cache/`, and `outputs/`.
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
- `run_synthesis.py`

## Pipeline Run Modes

Use `python -m synthesis.run_synthesis` from the repo root. Prefer `CUDA_VISIBLE_DEVICES=2,3` unless intentionally using another allocation. By default, **generation** uses **Amazon Bedrock** (`AWS_BEARER_TOKEN_BEDROCK` and `BEDROCK_GENERATION_MODEL` / `--generation-model`); **evaluation** still defaults to local vLLM with Qwen unless you pass other flags.

- Quick smoke run (fast sanity check, low sample count):
  `CUDA_VISIBLE_DEVICES=2,3 python -m synthesis.run_synthesis --task "Solve math word problems with constrained symbolic expressions." --dataset gsm_symbolic --min-accuracy 0.0 --min-syntax-rate 0.0 --max-iterations 1 --eval-sample-size 1 --eval-max-steps 256 --output-name smoke_gsm`
- Standard GSM-Symbolic synthesis run:
  `CUDA_VISIBLE_DEVICES=2,3 python -m synthesis.run_synthesis --task "Solve math word problems with constrained symbolic expressions." --dataset gsm_symbolic --min-accuracy 0.4 --min-syntax-rate 1.0 --max-iterations 5 --eval-sample-size 20 --eval-max-steps 900 --output-name gsm_main`
- Spider synthesis run (SQL), usually with higher per-step token budget:
  `CUDA_VISIBLE_DEVICES=2,3 python -m synthesis.run_synthesis --task "Generate executable SQL queries from natural language questions." --dataset spider --min-accuracy 0.6 --min-syntax-rate 0.95 --max-iterations 5 --eval-sample-size 20 --eval-max-steps 900 --eval-step-token-budget 8 --output-name spider_main`
- Spider run with explicit split manifest:
  `CUDA_VISIBLE_DEVICES=2,3 python -m synthesis.run_synthesis --task "Generate executable SQL queries from natural language questions." --dataset spider --spider-split-file <path/to/split.json> --spider-split-name train --min-accuracy 0.6 --min-syntax-rate 0.95 --max-iterations 5 --eval-sample-size 20 --output-name spider_split`
- SMILES synthesis run (all classes or class subset):
  `CUDA_VISIBLE_DEVICES=2,3 python -m synthesis.run_synthesis --task "Generate valid molecules in the requested class." --dataset smiles --smiles-classes acrylates,chain_extenders,isocyanates --smiles-samples-per-class 10 --min-accuracy 0.5 --min-syntax-rate 1.0 --max-iterations 5 --output-name smiles_main`
- Local generation with vLLM (override default Bedrock generation):
  `CUDA_VISIBLE_DEVICES=2,3 python -m synthesis.run_synthesis --task "Solve math word problems with constrained symbolic expressions." --dataset gsm_symbolic --generation-backend vllm --generation-model Qwen/Qwen2.5-Coder-7B-Instruct --eval-backend vllm --min-accuracy 0.4 --min-syntax-rate 1.0 --output-name vllm_run`
- Hosted generation defaults to **OpenAI** (`OPENAI_API_KEY`, model `gpt-5.4` or `OPENAI_GENERATION_MODEL`). **`gpt5.4`** in `run_all_tests.sh` uses OpenAI. **`opus4.7`** uses **Bedrock** (`AWS_BEARER_TOKEN_BEDROCK`, `BEDROCK_OPUS_MODEL`). The **`gemini-pro`** matrix profile is omitted until a partner wires it; pass `--generation-models gemini-pro` and set **`GEMINI_BEDROCK_MODEL`** when ready.
- Full repository test sweep:
  `bash run_all_tests.sh`
- `run_all_tests.sh` activates `/apps/conda/advayth2/envs/advayth2` by default and verifies RDKit import before starting the matrix. Partners using a different prefix should `export VAS_CONDA_ENV=/path/to/env`; `VAS_RDKIT_CONDA_ENV` remains as a legacy alias. The script prepends `CONDA_PREFIX/lib` to `LD_LIBRARY_PATH` so SciPy/transformers wheels resolve `libstdc++` correctly; Syncode needs **`mxeval`** with bundled **`data/`** — run **`bash environment/install_mxeval_into_env.sh`** once per env (see **`environment/README.md`**).

## Evaluation Expectations

- Always compare synthesized strategy performance against a CRANE baseline on the same model/split/sample settings before claiming success.
- Maintain high syntax/format validity while improving accuracy.
- Fixed-strategy GSM baselines should use the local CRANE GSM rows across `unconstrained`, `gcd`, `crane`, `itergen`, and `cars` so comparisons are row-aligned.
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
- Always update the `README.md` and `AGENTS.md` local to the folder you made changes in, and the global `README.md` and `AGENTS.md` for large changes. Occasionally scan the repo at the end of a request to ensure they are up-to-date.
- Under `synthesis/`, update the nearest subdirectory **`README.md`** / **`AGENTS.md`** when behavior or conventions change (see `synthesis/README.md` for the layout); do not add documentation inside vendored `synthesis/evaluate/syncode/syncode/` except via the root `evaluate/syncode/AGENTS.md` policy unless upgrading the vendor drop.
