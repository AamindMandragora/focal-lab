# AGENTS.md

## Scope

These instructions apply to work in this repository.

## Project Goal

Synthesize Dafny CSD strategies that verify, compile, and evaluate successfully, with the objective of outperforming the CRANE baseline on the same evaluation setup.

## Critical Prompting Rule

Do not include strategy guidance in synthesis prompts or task descriptions.

Allowed prompt content:
- Task objective.
- Tool signatures and formal contracts (preconditions, postconditions, types, ranges).

Disallowed prompt content:
- Recommendations about which tool to use.
- Preferred or forbidden strategy patterns.
- Baseline-comparison hints that imply structure.
- Procedural “NOTE” hints about when or why to apply a tool.

## Core Files

- `synthesis/verify/library/GeneratedCSD.dfy`
- `synthesis/verify/library/VerifiedAgentSynthesis.dfy`
- `synthesis/generate/generator.py`
- `synthesis/verify/verifier.py`
- `synthesis/verify/compiler.py`
- `synthesis/evaluate/feedback_loop.py`
- `synthesis/evaluate/evaluator.py`
- `run_synthesis.py`

## Evaluation Expectations

- Always compare synthesized strategy performance against a CRANE baseline on the same model/split/sample settings before claiming success.
- Maintain high syntax/format validity while improving accuracy.
- Keep benchmark-specific evaluation behavior in `synthesis/evaluate/benchmarks/*/eval_logic.py` and keep `synthesis/evaluate/evaluator.py` focused on orchestration/delegation.

## Performance Constraint

When touching parser validity logic, preserve DFA-mask-based validity checks (Syncode `DFAMaskStore`) for `ValidNextTokens` behavior. Do not introduce brute-force O(vocab) Lark parsing for per-step validity.

## Operational Defaults

- Prefer GPUs `2,3` for local runs unless intentionally using another allocation.
- Keep changes minimal and localized.
- Do not remove or alter formal contracts in Dafny files unless required by the task.
- Always update the `README.md` local to the folder you made changes in, and the global `README.md` and `AGENTS.md` for large changes.
