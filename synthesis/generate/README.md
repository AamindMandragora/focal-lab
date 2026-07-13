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
  - Supports local HuggingFace/vLLM generation plus OpenAI, Anthropic, direct Gemini, and explicit Bedrock hosted APIs.
- `prompts.py`
  - Prompt templates used for initial generation and iterative refinement.
  - Keep tool documentation here aligned with `synthesis/verify/library/README.md` and `VerifiedAgentSynthesis.dfy` when the strategy API changes.
  - Documents `helpers.AppendTaskGuidance(lm, guidance)` as a neutral API
    contract: call only at the start of a CSD, before generation helpers.
- `rationale.py`
  - Utilities for extracting or normalizing rationale sections embedded in strategy text.

## Rationale Claim Summaries

Evaluation refinement may include an attempt outcome ledger. When a prior
attempt's rationale is too long for that ledger, `StrategyGenerator` summarizes
the rationale with a small hosted model into one factual branch claim. Configure
this with `CSD_RATIONALE_SUMMARY_MODEL`; by default this uses the direct Gemini
API with `gemini-2.5-flash-lite`. Direct Gemini summary calls rotate through
`GEMINI_API_KEY_BACKUP_1`, `GEMINI_API_KEY_BACKUP_2`, ... on quota exhaustion
before falling back to Anthropic Haiku 4.5 (`claude-haiku-4-5`, override with
`CSD_RATIONALE_SUMMARY_FALLBACK_MODEL`). Set `CSD_RATIONALE_SUMMARY_BACKEND=openai`
only for explicit OpenAI summary experiments, or `CSD_RATIONALE_SUMMARY_BACKEND=off`
to skip the summarizer and include the full rationale. The final fallback is
full text rather than mechanical truncation so the causal hypothesis is not lost.

## Input and Output Contract

Input:

- Task description from CLI.
- Structured failure feedback from later stages.

Output:

- Strategy body text compatible with insertion into
  `synthesis/verify/library/GeneratedCSD.dfy`.

## API Backends

- **OpenAI** (`--generation-backend openai`): **`OPENAI_API_KEY`** (and optional **`OPENAI_BASE_URL`**). Default model **`gpt-5.4`** or **`OPENAI_GENERATION_MODEL`**. Synthesis author calls request reasoning effort **`xhigh`** by default; override with **`CSD_OPENAI_REASONING_EFFORT`** or **`OPENAI_GENERATION_REASONING_EFFORT`**, or set it to `off` only for intentional non-reasoning experiments. Used by the **`gpt5.5`** profile in `run_all_tests.py`.

- **Anthropic** (`--generation-backend anthropic`): **`ANTHROPIC_API_KEY`**. The default matrix profile is **`sonnet4.6`**, with optional **`ANTHROPIC_SONNET_MODEL`**; it uses adaptive thinking with `xhigh` effort by default. **`opus4.7`** remains a supported optional profile using **`ANTHROPIC_OPUS_MODEL`**.

- **Gemini** (`--generation-backend gemini`): **`GEMINI_API_KEY`** (or `GOOGLE_API_KEY`) and optional **`GEMINI_GENERATION_MODEL`**. On quota exhaustion, direct Gemini and Vertex API-key calls rotate through **`GEMINI_API_KEY_BACKUP_1`**, **`GEMINI_API_KEY_BACKUP_2`**, ... with no retry delay on the exhausted key. The **`gemini`** profile in `run_all_tests.py` uses the direct Gemini API with default model **`gemini-3-pro-preview`** and `CSD_GEMINI_THINKING_LEVEL=high`. Do not route Gemini through Bedrock.

- **Amazon Bedrock** (`--generation-backend bedrock`): retained only as a low-level explicit backend. The AWS Converse client path does not require `BEDROCK_BASE_URL`; the HTTP fallback derives the regional Bedrock runtime URL lazily when no explicit base URL is configured. Do not use Bedrock for matrix model ablations.

For explicit local smoke or infrastructure checks, use
**`--generation-backend huggingface`** or **`vllm`**. Do not use a small
local author for synthesis-quality runs or quality diagnosis.

The legacy `gemini-pro` matrix profile name is intentionally rejected because it used to mean a Bedrock-backed placeholder. Use the direct `gemini` profile instead.

## Important Constraint

This repository treats synthesis as a controlled study.
Prompt content must avoid strategy guidance, benchmark-specific answer hints,
and unmeasured heuristics. Neutral tool/API reference is allowed as contract
content as described in `AGENTS.md`.
