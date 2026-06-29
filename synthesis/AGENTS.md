# AGENTS.md — `synthesis/`

## Scope

Applies to the **`synthesis/`** Python package (pipeline implementation). Repository-wide rules live in **`../AGENTS.md`**.

## Invariants

- **Prompting:** synthesis prompts must follow the Critical Prompting Rule in the root `AGENTS.md` (task + formal contracts only; no strategy coaching).
- **Benchmarks:** keep task-specific scoring and prompts in **`evaluate/benchmarks/<name>/`**; keep **`evaluate/evaluator.py`** focused on orchestration.
- **Parser performance:** preserve DFA-mask (`DFAMaskStore`) validity paths; do not replace per-step validity with O(vocab) brute-force parsing.
- **Dafny contracts:** do not weaken or remove formal contracts in **`verify/library/`** unless the change is required and reviewed.
- **`run_synthesis.py` GSM-Symbolic:** the only supported data source is local CRANE-style JSONs (`--gsm-source-dir` auto-filled from vendored `legacy/CRANE/src/gsm_symbolic` when unset). HuggingFace loading has been removed; runs error out if no CRANE folder is resolvable.

## Subfolder guides

Each major subdirectory has its own **`README.md`** (human overview) and **`AGENTS.md`** (agent constraints). Prefer editing the leaf folder closest to your change.

Repository-wide archived manual assets live under **`../experiments/`** (not imported by this package).
