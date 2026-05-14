# AGENTS.md — `synthesis/generate/`

## Scope

Strategy **generation and refinement** prompts and orchestration.

## Rules

- Follow root **`AGENTS.md`** Critical Prompting Rule: neutral tool/API reference is allowed as contract content, but prompts must not add strategy guidance.
- When documenting `AppendTaskGuidance`, keep it as a neutral helper contract:
  append-only, first-call-wins, start-of-CSD placement only.
- Changes to **`prompts.py`** affect every synthesis run; keep diffs minimal and auditable.
- **`generator.py`** coordinates LLM calls and failure feedback; avoid embedding benchmark-specific hacks here (delegate via feedback shape or benchmark modules).

## See also

- **`README.md`** in this folder for file roles and I/O contract.
