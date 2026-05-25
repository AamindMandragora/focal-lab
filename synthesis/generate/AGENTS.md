# AGENTS.md — `synthesis/generate/`

## Scope

Strategy **generation and refinement** prompts and orchestration.

## Rules

- Follow root **`AGENTS.md`** Critical Prompting Rule: neutral tool/API reference is allowed as contract content, but prompts must not add strategy guidance.
- When documenting `AppendTaskGuidance`, keep it as a neutral helper contract:
  append-only, first-call-wins, start-of-CSD placement only.
- Changes to **`prompts.py`** affect every synthesis run; keep diffs minimal and auditable.
- **`generator.py`** coordinates LLM calls and failure feedback; avoid embedding benchmark-specific hacks here (delegate via feedback shape or benchmark modules).
- Matrix model ablations must use direct thinking-mode hosted profiles. Do not
  route matrix profiles through Bedrock or Bedrock-backed Gemini placeholders.
- Use the `gemini` matrix profile for direct Gemini API model ablations. The
  legacy `gemini-pro` profile name remains rejected because it historically
  referred to a Bedrock-backed placeholder.
- If prompt context needs rationale compression, use the rationale-summary path
  or preserve the full rationale. Do not replace rationale claims with
  character/word truncation in model-facing refinement context.

## Synthesizer defaults

- **`run_synthesis`**: default **`--generation-backend bedrock`** (Claude Opus via **`BEDROCK_OPUS_MODEL`**).
- **`run_all_tests.py`**: default **`--main-generation-model gemini`** for Phase~1 main-matrix and Ablation~A/B/D/E; default **`--generation-models sonnet4.6,gpt5.5`** for Ablation~C only (synthesizer-model study).

## See also

- **`README.md`** in this folder for file roles and I/O contract.
