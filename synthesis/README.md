# Synthesis Package

Each first-party subdirectory under `synthesis/` includes **`AGENTS.md`** (automation/agent constraints) alongside **`README.md`** (human-oriented overview) where applicable. Vendored Syncode under `evaluate/syncode/` uses a single **`AGENTS.md`** at that root; do not blanket nested vendor trees.

The `synthesis/` package is the operational center of the repository.
It provides an end-to-end loop that produces candidate CSD strategies, proves correctness properties in Dafny, compiles those strategies to executable Python modules, evaluates them on target benchmarks, and feeds failures back into the next generation attempt.

## Top-Level Modules

- `run_synthesis.py`
  - Main CLI entry point for iterative synthesis.
  - Configures models, thresholds, evaluation settings, and output layout.
  - **GSM-Symbolic:** the only supported data source is local CRANE-style JSONs. `--gsm-source-dir` auto-resolves to vendored `legacy/CRANE/src/gsm_symbolic` (or `$CRANE_GSM_SYMBOLIC_DIR`) when unset. HuggingFace loading has been removed; runs error out if no CRANE folder is resolvable.
  - Generation backends: local HuggingFace/vLLM and **OpenAI** (default for CLI). The generator still has a Bedrock-compatible fallback for targeted experiments, but the public matrix treats Bedrock profiles as experimental and rejects them.
  - Includes UCB/bandit helper-mask controls to constrain helper-call search space:
    `--adaptive-helper-mask`, `--helper-selection-policy`,
    `--helper-bandit-min-evals`, `--helper-bandit-top-k`,
    `--helper-bandit-ucb-c`, `--helper-bandit-explore-untried`.
  - Includes local-beam refinement controls:
    `--refinement-beam-size`, `--local-neighborhood-refinement`,
    `--max-local-edit-ratio`, `--beam-verify-candidates`.
  - Outer REx search-tree controls:
    `--rex-temperature` (Beta-prior temperature `C` for arm selection; default `2.0`).
- `project_defaults.py`
  - Centralized defaults for local paths (Dafny binary, CRANE/Spider resources, etc.).
- `failure_taxonomy.py`
  - Failure clustering and persistent ledger helpers for refinement prompts.

## Baseline and matrix entry points (under `evaluate/`)

- `run_legacy_fixed_strategy.py` — fixed baselines via legacy repos + vendored SynCode.
- `run_reference_strategy.py` — compile/eval verified Dafny reference strategies.
- `export_baseline_json.py` — minimal baseline JSON export helper.

## Scripts

- `scripts/reevaluate_compiled_csd.py` — re-evaluate a compiled strategy (used by `run_all_tests.py`).

## Stage Subpackages

- `generate/`
  - Prompt construction and strategy body generation/refinement.
- `verify/`
  - Dafny verification and Dafny-to-Python compilation wrappers.
  - `verify/reference/` holds verified example strategies (CRANE / IterGen / CARS-style); see `verify/reference/README.md`.
- `evaluate/`
  - Runtime environment setup, benchmark evaluation, parser integration, and feedback-loop orchestration.
  - Captures CSD-authored `AppendTaskGuidance` prompt guidance in evaluation
    feedback so refinement can compare guidance choices against metrics.

## Canonical Stage Flow

The package is intentionally structured to mirror the pipeline order:

1. `generate`
2. `verify`
3. `evaluate`

Compile is implemented under `verify` because compilation is only valid after verification and shares Dafny-specific lifecycle code.

## Artifact Ownership

- Successful and failed run artifacts are written under `outputs/generated/`.
- Baseline experiment artifacts are written under `outputs/baselines/`.
- `synthesis/` itself contains implementation, not experiment outputs.

## Path Overrides

Common filesystem/tool paths can be overridden via CLI flags or environment variables:

- `--output-dir` or `CSD_OUTPUT_DIR`
- `--baseline-output-dir` or `CSD_BASELINE_OUTPUT_DIR`
- `--grammars-dir` or `CSD_GRAMMARS_DIR`
- `--dafny-path` or `DAFNY_PATH`
- `DAFNY_EXTRA_PATH` (colon-separated PATH entries for Dafny subprocesses)
- `VERIFIED_AGENT_SYNTHESIS_DFY` or `DAFNY_PROOFS_DIR` (override proof include source)
- `CSD_SYNCODE_DIR` (vendored Syncode root)
- `SPIDER_DATA_DIR`, `SPIDER_DB_DIR`, `SPIDER_TABLES_JSON`
- `SPIDER_EVAL_DIR` / `SPIDER_EVAL_PY` (Spider evaluator location)
- `SMILES_DATA_DIR`, `SMILES_GRAMMAR_DIR`
- `CSD_JSON_GRAMMAR_PATH` (JSON grammar for smoke-test runner)
- `OPENAI_API_KEY` / `OPENAI_GENERATION_MODEL` for OpenAI (default `--generation-backend openai` in `run_synthesis`, model `gpt-5.4` unless overridden).
- `AWS_BEARER_TOKEN_BEDROCK` and Bedrock model ids when using `--generation-backend bedrock`.

## Design Philosophy

- Keep benchmark-specific behavior in benchmark packages, not in generic pipeline code.
- Keep parser/runtime performance guarantees explicit (Syncode DFA masks, caching, parser factories).
- Keep each synthesis attempt auditable through saved reports and run-local artifacts.
