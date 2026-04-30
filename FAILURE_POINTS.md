# Failure Points In Verified Agent Synthesis

## Definition Of Failure
A run is considered a failure if any of the following occurs:

- Subpar evaluation: accuracy / format-rate / syntax-rate misses configured thresholds.
- Serious pipeline failure: generation, verification, runtime, or evaluation halts before a valid CSD is accepted within allotted attempts.
- Search failure: no acceptable strategy is produced by `max_iterations` / search-attempt limits.

## High-Risk Failure Points

### 1. Prompt ↔ Helper Surface Drift
- Location: `generation/prompts.py`, `generation/csd/VerifiedAgentSynthesis.py`
- Symptom: generated code calls non-existent helpers or wrong signatures.
- Impact: structural rejection during generation; zero useful candidates.

### 2. Structural Validator Overconstraint / Misconstraint
- Location: `generation/generator.py` (`_structural_issue`)
- Symptom: good strategies rejected for policy mismatch; repeated generation failures.
- Impact: run dies at generation despite potentially viable CSDs.

### 3. Transpiler Named-Return Mapping Drift
- Location: `verification/transpiler/transpiler.py` (`_RETURN_NAME_OVERRIDES`)
- Symptom: tuple-return helpers emitted as single-return Dafny methods.
- Impact: verification/type errors despite valid Python helper definitions.

### 4. Dafny Contract Strictness Mismatch
- Location: `generation/csd/VerifiedAgentSynthesis.py`
- Symptom: helper contracts too strong for implementation or for transpiled proof obligations.
- Impact: verification fails even with syntactically correct generation.

### 5. Delimiter Policy Conflicts Across Modes
- Location: `generation/generator.py`, `synthesis/presets.py`, shell env vars
- Symptom: Spider/GSM policy flags contradict structural checks.
- Impact: systematic rejections (e.g., closure policy contradictions).

### 6. Budget-Logic Deadlocks In Generated Strategies
- Location: generated CSD bodies (`outputs/*/GeneratedCSD.py`)
- Symptom: loops break early, or never produce/close answer span.
- Impact: format/syntax failures or empty extracted answer.

### 7. Parser-State Misuse
- Location: generated strategy + helper wrappers
- Symptom: direct parser calls on wrong prefix or wrong completion checks.
- Impact: malformed constrained spans; low syntax-rate.

### 8. Runtime Helper Semantics Changes
- Location: `generation/csd/VerifiedAgentSynthesis.py`
- Symptom: helper behavior changes (masking, biasing, delimiter handling) without synchronized prompts/tests.
- Impact: degraded generation quality and unstable evaluation.

### 9. Evaluation Extraction Assumption Mismatch
- Location: `evaluation/evaluator.py`, task-specific metrics/extractors
- Symptom: generated output is semantically reasonable but extraction/grading expects a different span pattern.
- Impact: low measured accuracy despite plausible outputs.

### 10. LM/Tokenizer State Leakage Across Evaluations
- Location: `evaluation/common/environment.py`, `evaluation/common/model_utils.py`, evaluator lifecycle
- Symptom: one sample/run influences later samples via retained state.
- Impact: noisy or biased metrics, poor reproducibility.

### 11. Grammar Coverage Gaps
- Location: `utils/grammars/*.lark` (especially SQL)
- Symptom: valid target answers not representable or hard to represent.
- Impact: constrained decoder cannot emit needed outputs; accuracy ceiling.

### 12. Search Diversity Collapse
- Location: generation prompting + refinement loop
- Symptom: model repeats one brittle strategy template across attempts.
- Impact: no recovery from local minima; repeated low-quality failures.

### 13. Refinement-Loop Self-Poisoning
- Location: synthesis feedback loop + repair prompts
- Symptom: bad repair instructions accumulate and further narrow search incorrectly.
- Impact: attempt budget consumed without meaningful improvement.

### 14. Run-Orchestration / Artifact Handling Errors
- Location: shell scripts, `outputs/latest`, run artifact resolution
- Symptom: stale run picked up, partial output dirs, misreported outcomes.
- Impact: misleading diagnostics and wrong decisions.

### 15. Resource / Environment Instability
- Location: model loading + GPU/CPU path + offline cache settings
- Symptom: hanging runs, OOM/fallback churn, inconsistent runtime behavior.
- Impact: incomplete runs and non-comparable evaluation results.

## Early-Warning Checks To Run Per Change

- `python verification/transpiler/transpiler.py generation/csd/VerifiedAgentSynthesis.py`
- `python -m py_compile generation/*.py synthesis/*.py verification/transpiler/*.py`
- Focused tests on changed surfaces (prompt/generator/transpiler/helper contracts).
- One bounded end-to-end smoke run per dataset mode affected.
- Inspect `failure_report.json` for repeated structural-rejection motifs before broad reruns.

## Practical Triage Order When A Run Fails

1. Check `failed_at` in `failure_report.json` (`generation` vs `verification` vs `runtime` vs `evaluation`).
2. If `generation`: inspect `_structural_issue` rejection text and helper-name drift first.
3. If `verification`: check transpiler return mappings and helper contract shape.
4. If `runtime/evaluation`: inspect extracted span behavior, grammar fit, and evaluator assumptions.
5. Only then tune prompt/search parameters.
