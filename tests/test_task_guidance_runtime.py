from synthesis.evaluate.benchmarks.common.model_utils import _TaskGuidanceState, _TensorizedLMBase
from synthesis.evaluate.evaluator import EvaluationResult
from synthesis.generate import prompts


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


def test_lm_append_task_guidance_coerces_sequence_guidance():
    class FakeSeq:
        def __init__(self, text):
            self._text = text

        def __len__(self):
            return len(self._text)

        def __getitem__(self, index):
            return self._text[index]

        def __str__(self):
            return "<fake-seq>"

    lm = _TensorizedLMBase(object(), None, [], [], logits_device="cpu")
    lm.instruction_text = "SYSTEM"

    lm.AppendTaskGuidance(FakeSeq("Sequence guidance."))

    assert lm.task_guidance == "Sequence guidance."
    assert "Sequence guidance." in lm.instruction_text


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


def test_prompt_api_documents_append_task_guidance_start_only():
    tool_reference = prompts.TOOL_REFERENCE

    assert "helpers.AppendTaskGuidance(lm, guidance);" in tool_reference
    assert "call only at the start of the CSD" in tool_reference


def test_verified_examples_place_guidance_before_generation_helpers():
    example_index = prompts.VERIFIED_EXAMPLES.index("helpers.AppendTaskGuidance")
    first_generation_index = min(
        index
        for index in [
            prompts.VERIFIED_EXAMPLES.find("helpers.UnconstrainedStep", example_index),
            prompts.VERIFIED_EXAMPLES.find("helpers.ConstrainedStep", example_index),
            prompts.VERIFIED_EXAMPLES.find("helpers.UnconstrainedChunk", example_index),
            prompts.VERIFIED_EXAMPLES.find("helpers.ConstrainedSymbol", example_index),
        ]
        if index != -1
    )

    assert example_index < first_generation_index
