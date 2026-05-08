# Evaluation Grammars

This directory stores Lark grammar files used to constrain token generation during synthesis evaluation.

## What These Grammars Control

- Which next tokens are valid under the current parser state.
- How syntax validity is checked for produced constrained spans.
- Dataset-specific structural contracts (math spans, SQL forms, molecule syntax, etc.).

## Active Grammar Families

- GSM-Symbolic grammars (`gsm*.lark`, `math.lark`)
- SQL grammar (`sql.lark`)
- SMILES grammar (`smiles.lark`)
- Utility grammar variants (`json*.lark`) for parser/runtime support

## Performance Note

These grammars are consumed through Syncode-backed DFA mask stores.
That path is essential for fast valid-next-token lookup and should remain the default runtime mode.
