# CSD Task Guidance Tool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an append-only, first-call-wins `AppendTaskGuidance` CSD helper whose guidance is applied at the start of each evaluation example and reported back in refinement feedback.

**Architecture:** Add a Dafny LM extern plus `CSDHelpers` wrapper, implement the runtime prompt mutation in the shared LM base, capture accepted guidance in evaluation records, and expose neutral API docs plus verified examples in synthesis prompts. The helper is zero-cost, append-only, and intended only as a first executable action after output initialization.

**Tech Stack:** Dafny contracts compiled to Python, Python evaluation runtime, synthesis prompt templates, focused `python3` validation tests.

---

### Task 1: Add Failing Runtime Guidance Tests

**Files:**
- Create: `tests/test_task_guidance_runtime.py`
- Test: `tests/test_task_guidance_runtime.py`

- [ ] **Step 1: Write tests for first-call-wins runtime state**

Create `tests/test_task_guidance_runtime.py` with:

```python
from synthesis.evaluate.benchmarks.common.model_utils import _TaskGuidanceState


def test_append_task_guidance_appends_once():
    state = _TaskGuidanceState()
    prompt = "SYSTEM\n"

    updated = state.append(prompt, "Avoid arithmetic slips.")

    assert updated == "SYSTEM\n\nAdditional task guidance from CSD:\nAvoid arithmetic slips.\n"
    assert state.accepted_guidance == "Avoid arithmetic slips."


def test_append_task_guidance_first_call_wins():
    state = _TaskGuidanceState()
    first = state.append("SYSTEM", "First guidance.")
    second = state.append(first, "Second guidance.")

    assert second == first
    assert state.accepted_guidance == "First guidance."


def test_append_task_guidance_empty_is_noop():
    state = _TaskGuidanceState()

    updated = state.append("SYSTEM", "   ")

    assert updated == "SYSTEM"
    assert state.accepted_guidance is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. /opt/anaconda/bin/python -m pytest tests/test_task_guidance_runtime.py -q`

Expected: FAIL during import because `_TaskGuidanceState` does not exist yet.

### Task 2: Implement Runtime Guidance State

**Files:**
- Modify: `synthesis/evaluate/benchmarks/common/model_utils.py`
- Test: `tests/test_task_guidance_runtime.py`

- [ ] **Step 1: Add `_TaskGuidanceState` and LM methods**

In `synthesis/evaluate/benchmarks/common/model_utils.py`, add a small state class near `_TensorizedLMBase`:

```python
class _TaskGuidanceState:
    """First-call-wins prompt guidance appended by generated CSDs."""

    MAX_GUIDANCE_CHARS = 1200
    HEADER = "Additional task guidance from CSD:"

    def __init__(self) -> None:
        self.accepted_guidance: str | None = None

    def reset(self) -> None:
        self.accepted_guidance = None

    def append(self, instruction_text: str, guidance: object) -> str:
        if self.accepted_guidance is not None:
            return instruction_text
        text = self._coerce_guidance(guidance)
        if not text:
            return instruction_text
        self.accepted_guidance = text
        return f"{instruction_text}\n\n{self.HEADER}\n{text}\n"

    def _coerce_guidance(self, guidance: object) -> str:
        text = str(guidance).strip()
        if not text:
            return ""
        return text[: self.MAX_GUIDANCE_CHARS]
```

In `_TensorizedLMBase.__init__`, initialize `self._task_guidance = _TaskGuidanceState()`.

Add:

```python
    def ResetTaskGuidance(self):
        self._task_guidance.reset()

    def AppendTaskGuidance(self, guidance):
        self.instruction_text = self._task_guidance.append(self.instruction_text, guidance)

    @property
    def task_guidance(self) -> str | None:
        return self._task_guidance.accepted_guidance
```

- [ ] **Step 2: Run runtime guidance tests**

Run: `PYTHONPATH=. /opt/anaconda/bin/python -m pytest tests/test_task_guidance_runtime.py -q`

Expected: PASS.

### Task 3: Add Failing Feedback Tests

**Files:**
- Modify: `tests/test_task_guidance_runtime.py`
- Test: `tests/test_task_guidance_runtime.py`

- [ ] **Step 1: Add EvaluationResult feedback test**

Append:

```python
from synthesis.evaluate.evaluator import EvaluationResult


def test_feedback_summary_reports_prompt_guidance():
    result = EvaluationResult(
        success=True,
        accuracy=0.5,
        contains_delimiters=True,
        syntax_rate=1.0,
        num_examples=2,
        num_correct=1,
        total_time_seconds=3.0,
        sample_outputs=[{"task_guidance": "Avoid arithmetic slips."}],
        task_guidance=["Avoid arithmetic slips."],
    )

    summary = result.get_feedback_summary()

    assert "Prompt guidance used by this attempt:" in summary
    assert "Avoid arithmetic slips." in summary
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. /opt/anaconda/bin/python -m pytest tests/test_task_guidance_runtime.py::test_feedback_summary_reports_prompt_guidance -q`

Expected: FAIL because `EvaluationResult` has no `task_guidance` field.

### Task 4: Capture Guidance in Evaluation Results

**Files:**
- Modify: `synthesis/evaluate/evaluator.py`
- Modify: `synthesis/evaluate/benchmarks/gsm_symbolic/generation.py`
- Test: `tests/test_task_guidance_runtime.py`

- [ ] **Step 1: Add result field and feedback block**

In `EvaluationResult`, add:

```python
    task_guidance: List[str] = field(default_factory=list)
```

In `get_feedback_summary`, after the early/accuracy metadata and before task-specific metrics, add:

```python
        if self.task_guidance:
            lines.extend(["", "Prompt guidance used by this attempt:"])
            for guidance in self.task_guidance:
                lines.append(f"  - {guidance}")
```

- [ ] **Step 2: Reset and capture guidance per example**

In `synthesis/evaluate/benchmarks/gsm_symbolic/generation.py`, after assigning `lm.instruction_text`, call:

```python
    if hasattr(lm, "ResetTaskGuidance"):
        lm.ResetTaskGuidance()
```

After `helper_trace` is created, capture:

```python
    task_guidance = getattr(lm, "task_guidance", None)
    if task_guidance:
        helper_trace.append({"helper": "AppendTaskGuidance", "detail": task_guidance})
```

In `synthesis/evaluate/evaluator.py`, when creating each sample output, add:

```python
                        "task_guidance": getattr(env.get("lm"), "task_guidance", None),
```

Before returning `EvaluationResult`, compute:

```python
            task_guidance = sorted({
                s.get("task_guidance")
                for s in sample_outputs
                if s.get("task_guidance")
            })
```

Pass `task_guidance=task_guidance` to each `EvaluationResult` return path that has `sample_outputs`.

- [ ] **Step 3: Run feedback tests**

Run: `PYTHONPATH=. /opt/anaconda/bin/python -m pytest tests/test_task_guidance_runtime.py -q`

Expected: PASS.

### Task 5: Add Dafny Helper and Prompt Contract

**Files:**
- Modify: `synthesis/verify/library/VerifiedAgentSynthesis.dfy`
- Modify: `synthesis/generate/prompts.py`
- Test: `tests/test_task_guidance_runtime.py`

- [ ] **Step 1: Add failing prompt/template tests**

Append:

```python
from synthesis.generate import prompts


def test_prompt_api_documents_append_task_guidance_start_only():
    system_prompt = prompts.SYSTEM_PROMPT

    assert "helpers.AppendTaskGuidance(lm, guidance);" in system_prompt
    assert "call only at the start of the CSD" in system_prompt


def test_verified_examples_place_guidance_before_generation_helpers():
    example_index = prompts.VERIFIED_EXAMPLES.index("helpers.AppendTaskGuidance")
    first_generation_index = min(
        index for index in [
            prompts.VERIFIED_EXAMPLES.find("helpers.UnconstrainedStep", example_index),
            prompts.VERIFIED_EXAMPLES.find("helpers.ConstrainedStep", example_index),
            prompts.VERIFIED_EXAMPLES.find("helpers.UnconstrainedChunk", example_index),
            prompts.VERIFIED_EXAMPLES.find("helpers.ConstrainedSymbol", example_index),
        ] if index != -1
    )

    assert example_index < first_generation_index
```

Run: `PYTHONPATH=. /opt/anaconda/bin/python -m pytest tests/test_task_guidance_runtime.py::test_prompt_api_documents_append_task_guidance_start_only tests/test_task_guidance_runtime.py::test_verified_examples_place_guidance_before_generation_helpers -q`

Expected: FAIL because prompt docs/examples do not mention the helper yet.

- [ ] **Step 2: Add Dafny extern and helper wrapper**

In `class LM`, add:

```dafny
    method {:extern} {:axiom} AppendTaskGuidance(guidance: string)
      requires ValidTokensIdsLogits()
      ensures ValidTokensIdsLogits()
```

In `class CSDHelpers`, add:

```dafny
    method AppendTaskGuidance(lm: LM, guidance: string)
      requires lm.ValidTokensIdsLogits()
      ensures lm.ValidTokensIdsLogits()
      ensures cost == old(cost)
    {
      lm.AppendTaskGuidance(guidance);
    }
```

- [ ] **Step 3: Update synthesis prompt API reference**

In `SYSTEM_PROMPT`, add `helpers.AppendTaskGuidance(lm, guidance);` to helper methods and add a neutral note:

```text
`AppendTaskGuidance` appends a CSD-chosen guidance block to the evaluator's existing task prompt. It is append-only, first-call-wins, and costs 0. Call only at the start of the CSD, after output initialization and before the first LM generation helper. Do not use it as a mid-generation control action.
```

- [ ] **Step 4: Add verified example demonstrating first-line placement**

Add a verified example where the method body initializes outputs, calls `helpers.AppendTaskGuidance(...)`, and then enters ordinary delimiter-triggered decoding. The example should place no generation helper before `AppendTaskGuidance`.

- [ ] **Step 5: Run tests**

Run: `PYTHONPATH=. /opt/anaconda/bin/python -m pytest tests/test_task_guidance_runtime.py -q`

Expected: PASS.

### Task 6: Documentation and Validation

**Files:**
- Modify: `synthesis/README.md`
- Modify: `synthesis/generate/README.md`
- Modify: `synthesis/evaluate/README.md`
- Modify: `synthesis/verify/library/README.md`
- Test: py_compile / pytest

- [ ] **Step 1: Document the new helper near changed surfaces**

Update the nearest READMEs to mention:

```text
Generated CSDs may call `helpers.AppendTaskGuidance(lm, guidance)` as the first action after output initialization. The runtime appends the first non-empty guidance block to the evaluator prompt, ignores later calls, and includes the accepted guidance in evaluation feedback.
```

- [ ] **Step 2: Run focused verification**

Run:

```bash
PYTHONPATH=. /opt/anaconda/bin/python -m pytest tests/test_task_guidance_runtime.py -q
python3 -m py_compile synthesis/generate/prompts.py synthesis/evaluate/evaluator.py synthesis/evaluate/benchmarks/common/model_utils.py synthesis/evaluate/benchmarks/gsm_symbolic/generation.py
```

Expected: all tests pass and `py_compile` exits 0.

- [ ] **Step 3: Review diff**

Run: `git diff --stat` and `git diff --check`.

Expected: no whitespace errors and changes limited to the planned files.
