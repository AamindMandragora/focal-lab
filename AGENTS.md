# AGENTS.md

## Scope

These instructions apply to work in this repository.

## Project Goal

Synthesize Dafny CSD strategies that verify, compile, and evaluate successfully, with the objective of outperforming the CRANE baseline on the same evaluation setup.

## Critical Prompting Rule

Do not include strategy guidance in synthesis prompts or task descriptions.

Allowed prompt content:
- Task objective.
- Tool signatures and formal contracts (preconditions, postconditions, types, ranges).

Disallowed prompt content:
- Recommendations about which tool to use.
- Preferred or forbidden strategy patterns.
- Baseline-comparison hints that imply structure.
- Procedural “NOTE” hints about when or why to apply a tool.

## Key Paths

- Dafny binary: `dafny/dafny`
- OpenAI API key: `synthesis/.env`

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

Use `python -m synthesis.run_synthesis` from the repo root. Prefer `CUDA_VISIBLE_DEVICES=2,3` unless intentionally using another allocation. By default, **generation** uses OpenAI `gpt-5.4` (`OPENAI_API_KEY`); **evaluation** still defaults to local vLLM with Qwen unless you pass other flags.

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
- Local generation with vLLM (override default OpenAI generation):
  `CUDA_VISIBLE_DEVICES=2,3 python -m synthesis.run_synthesis --task "Solve math word problems with constrained symbolic expressions." --dataset gsm_symbolic --generation-backend vllm --generation-model Qwen/Qwen2.5-Coder-7B-Instruct --eval-backend vllm --min-accuracy 0.4 --min-syntax-rate 1.0 --output-name vllm_run`
- API generation with a non-default model (default generation is already OpenAI `gpt-5.4`):
  `python -m synthesis.run_synthesis --task "Solve math word problems with constrained symbolic expressions." --dataset gsm_symbolic --generation-backend openai --generation-model <api-model> --eval-backend vllm --min-accuracy 0.4 --min-syntax-rate 1.0 --output-name api_gen_run`
- Full repository test sweep:
  `bash run_all_tests.sh`

## Evaluation Expectations

- Always compare synthesized strategy performance against a CRANE baseline on the same model/split/sample settings before claiming success.
- Maintain high syntax/format validity while improving accuracy.
- Keep benchmark-specific evaluation behavior in `synthesis/evaluate/benchmarks/*/eval_logic.py` and keep `synthesis/evaluate/evaluator.py` focused on orchestration/delegation.

## Performance Constraint

When touching parser validity logic, preserve DFA-mask-based validity checks (Syncode `DFAMaskStore`) for `ValidNextTokens` behavior. Do not introduce brute-force O(vocab) Lark parsing for per-step validity.

## Operational Defaults

- Prefer GPUs `2,3` for local runs unless intentionally using another allocation.
- Keep changes minimal and localized.
- Do not remove or alter formal contracts in Dafny files unless required by the task.
- Do not create, edit, delete, or commit files under `paper/` unless the user explicitly requests changes there (the paper tree is out of scope for routine agent work).
- When asked to modify files under `paper/`, do not run paper compilation checks; after making the requested edits, always automatically proceed with `git add`, `git commit`, and `git push` within the subdirectory to update Overleaf.
- When using adaptive helper masking or beam refinement, keep selection rules empirical/contract-based (measured metrics, verifier checks), not heuristic strategy advice in prompts.
- For bandit-style helper selection, keep exploration/exploitation policy in pipeline code/CLI knobs (e.g., UCB parameters), not in synthesis prompt prose.
- Always update the `README.md` local to the folder you made changes in, and the global `README.md` and `AGENTS.md` for large changes.
- Under `synthesis/`, update the nearest subdirectory **`README.md`** / **`AGENTS.md`** when behavior or conventions change (see `synthesis/README.md` for the layout); do not add documentation inside vendored `synthesis/evaluate/syncode/syncode/` except via the root `evaluate/syncode/AGENTS.md` policy unless upgrading the vendor drop.
