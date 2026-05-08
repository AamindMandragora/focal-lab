# Common Benchmark Utilities

`benchmarks/common` provides reusable building blocks shared by all benchmark adapters.

## Responsibilities

- Runtime model loading and wrapper construction.
- Parser construction utilities bridging Dafny parser interfaces to Syncode/Lark internals.
- Shared caching and performance-sensitive infrastructure used across benchmarks.

## Key Modules

- `model_utils.py`
  - Backend-aware model/tokenizer setup and runtime object creation.
- `parser_utils.py`
  - Canonical parser factory implementation using Syncode DFA mask stores.

## Why This Exists

Without a shared common layer, benchmark packages would drift into copy-pasted runtime code.
Centralizing these utilities makes parser/runtime behavior consistent and easier to optimize safely.
