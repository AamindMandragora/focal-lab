# Verify Library (Dafny)

This directory contains the Dafny files that define the formal substrate for synthesized strategies.
These files are the foundation for both verification and runtime compilation.

## Files

- `GeneratedCSD.dfy`
  - Template file containing insertion markers where generated strategy logic is placed.
  - Treated as a reusable scaffold; synthesis should not permanently overwrite template semantics.
- `VerifiedAgentSynthesis.dfy`
  - Core verified definitions, contracts, helper APIs, and proof obligations used by generated strategies.

## Role in the Pipeline

1. Generation creates a candidate strategy body.
2. The body is injected into the template contract context.
3. Dafny verifies the assembled program.
4. Verified code is compiled to Python for evaluation.

## Editing Guidance

- Changes here can invalidate synthesis assumptions and proof obligations.
- Treat edits as high-impact; update generation prompts and evaluation expectations when contracts change.
- Keep method signatures and contract semantics stable unless intentionally evolving the synthesis interface.
