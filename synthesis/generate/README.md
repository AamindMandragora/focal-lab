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
  - Supports local HuggingFace/vLLM generation and API backends: OpenAI, Anthropic, Gemini, and Bedrock.
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

Generation API credentials are read from backend-specific environment variables:

- `OPENAI_API_KEY` for `--generation-backend openai`
- `ANTHROPIC_API_KEY` for `--generation-backend anthropic`
- `GEMINI_API_KEY` or `GOOGLE_API_KEY` for `--generation-backend gemini`
- `AWS_BEARER_TOKEN_BEDROCK` for `--generation-backend bedrock`

For Bedrock, pass a Bedrock model id via `--generation-model`; `BEDROCK_BASE_URL` can override the runtime endpoint, otherwise the generator derives it from `AWS_REGION` / `AWS_DEFAULT_REGION` and defaults to `us-east-1`.

## Important Constraint

This repository treats synthesis as a controlled study.
Prompt content must avoid hidden strategy coaching and should remain limited to allowed task/tool context as described in `AGENTS.md`.
