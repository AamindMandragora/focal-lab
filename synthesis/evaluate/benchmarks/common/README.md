# Common Benchmark Utilities

`benchmarks/common` provides reusable building blocks shared by all benchmark adapters.

## Responsibilities

- Runtime model loading and wrapper construction.
- Parser construction utilities bridging Dafny parser interfaces to Syncode/Lark internals.
- Shared caching and performance-sensitive infrastructure used across benchmarks.

## Key Modules

- `delimiter_grammar.py` / `delimited_completion.py`
  - Adapt Lark grammars and raw decoder output for tier-1 `<<` … `>>` spans (GCD / IterGen / CARS).
- `delimited_output.py`
  - Shared `<< >>` extraction for GSM, Spider, and SMILES scoring.
- `model_utils.py`
  - Backend-aware model/tokenizer setup and runtime object creation.
  - Owns first-call-wins task guidance state for CSD-authored prompt guidance.
- `parser_utils.py`
  - Canonical parser factory implementation using Syncode DFA mask stores.

## Why This Exists

Without a shared common layer, benchmark packages would drift into copy-pasted runtime code.
Centralizing these utilities makes parser/runtime behavior consistent and easier to optimize safely.
