# Evaluate Stage

The evaluate stage executes compiled strategies on benchmark tasks and returns structured metrics used by the synthesis feedback loop.

## Responsibilities

- Build runtime environment around compiled Dafny output.
- Load benchmark datasets and grammar resources.
- Run constrained decoding evaluation for each sample.
- Compute benchmark metrics and synthesis gate checks.
- Emit rich diagnostics used for strategy refinement.

## Main Components

- `evaluator.py`
  - Core sample evaluation loop, scoring logic, syntax checks, and answer extraction.
- `feedback_loop.py`
  - Generate/verify/compile/evaluate orchestration with iterative refinement.
- `runner.py`
  - Runtime helper paths used by local smoke/runtime routines.
- `parser_utils.py`
  - Compatibility wrapper re-exporting canonical parser utilities.
- `benchmarks/`
  - Dataset-specific modules (GSM-Symbolic, SQL Spider, SMILES).
- `grammars/`
  - Lark grammar definitions used by constrained decoding.
- `syncode/`
  - Vendored Syncode dependency for DFA mask store + parser internals.

## Runtime Constraints

- The parser path depends on Syncode DFA-mask caching for practical performance.
- Evaluation backends currently support runtime modes that provide token-level control (`huggingface`, `vllm`).
- Output artifacts from this stage are saved under per-run `results/` folders inside repo-root `generated/`.
