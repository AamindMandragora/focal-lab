# Generate Stage

The generate stage is responsible for producing candidate Dafny strategy bodies that can be inserted into the strategy template and passed through formal verification.

## Responsibilities

- Define and maintain synthesis prompts.
- Generate initial strategy candidates from the user task.
- Refine strategies when verification, compilation, runtime, or evaluation fails.
- Preserve model-facing rationale blocks when required by project conventions.

## Key Files

- `generator.py`
  - Main generation/refinement orchestration.
  - Supports local HuggingFace/vLLM generation plus OpenAI and Amazon Bedrock hosted APIs.
- `prompts.py`
  - Prompt templates used for initial generation and iterative refinement.
  - Keep tool documentation here aligned with `synthesis/verify/library/README.md` and `VerifiedAgentSynthesis.dfy` when the strategy API changes.
- `rationale.py`
  - Utilities for extracting or normalizing rationale sections embedded in strategy text.

## Input and Output Contract

Input:

- Task description from CLI.
- Structured failure feedback from later stages.

Output:

- Strategy body text compatible with insertion into
  `synthesis/verify/library/GeneratedCSD.dfy`.

## API Backends

- **OpenAI** (`--generation-backend openai`): **`OPENAI_API_KEY`** (and optional **`OPENAI_BASE_URL`**). Default model **`gpt-5.4`** or **`OPENAI_GENERATION_MODEL`**. Used by the **`gpt5.4`** profile in `run_all_tests.sh`.

- **Amazon Bedrock** (`--generation-backend bedrock`): **`AWS_BEARER_TOKEN_BEDROCK`** and a Bedrock model id via **`--generation-model`** or **`BEDROCK_GENERATION_MODEL`**. Used by the **`opus4.7`** profile (see **`BEDROCK_OPUS_MODEL`** in `run_all_tests.sh`).

For local runs use **`--generation-backend huggingface`** or **`vllm`**.

Direct Gemini / Vertex APIs are not wired in the generator; Gemini matrix profiling is partner-owned (`GEMINI_BEDROCK_MODEL` placeholder in `run_all_tests.sh` when `gemini-pro` is enabled).

## Important Constraint

This repository treats synthesis as a controlled study.
Prompt content must avoid hidden strategy coaching and should remain limited to allowed task/tool context as described in `AGENTS.md`.
