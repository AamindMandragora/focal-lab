from types import SimpleNamespace

import pytest

from synthesis.evaluate.feedback_loop import SynthesisExhaustionError, SynthesisPipeline
from synthesis.generate.generator import StrategyGenerator
try:
    from synthesis.verify.verifier import VerificationError, VerificationResult
except ModuleNotFoundError:
    from synthesis.verifier import VerificationError, VerificationResult


TASK = "Solve the fake sandbox task using the exact task context."
STRATEGY = """// CSD_RATIONALE_BEGIN
// Fake strategy used only to exercise warm-resume refinement.
// CSD_RATIONALE_END
var sandboxValue := 0;
"""


def test_warm_resume_refinement_keeps_real_task_context(tmp_path):
    """A replayed strategy must refine against its real task, not `Unknown task`."""
    captured_user_prompts: list[str] = []
    generator = StrategyGenerator(backend="openai", api_key="fake", device="cpu")

    def fake_generate(_system_prompt: str, user_prompt: str) -> str:
        captured_user_prompts.append(user_prompt)
        return STRATEGY

    generator._generate_text = fake_generate
    evaluator = SimpleNamespace(
        model_name="fake-eval-model",
        dataset_name="fake-dataset",
        max_steps=1,
        step_token_budget=1,
    )
    verifier = SimpleNamespace(
        verify=lambda _code: VerificationResult(
            success=False,
            errors=[
                VerificationError(
                    file="Fake.dfy",
                    line=1,
                    column=1,
                    message="fake verification failure",
                )
            ],
        )
    )
    compiler = SimpleNamespace(
        dafny_path="unused-in-sandbox",
        timeout=1,
        extra_args=[],
    )
    pipeline = SynthesisPipeline(
        evaluator=evaluator,
        generator=generator,
        verifier=verifier,
        compiler=compiler,
        max_iterations=2,
        output_dir=tmp_path,
        save_reports=False,
        adaptive_helper_mask=False,
        refinement_beam_size=1,
    )

    with pytest.raises(SynthesisExhaustionError):
        pipeline.synthesize(
            task_description=TASK,
            output_name="fake_warm_resume",
            initial_strategy_code=STRATEGY,
            initial_attempt_offset=9,
        )

    assert captured_user_prompts, "the fake cycle never reached refinement"
    assert all(TASK in prompt for prompt in captured_user_prompts)
    assert all("Unknown task" not in prompt for prompt in captured_user_prompts)
