"""Generation profile resolution and eval-model slug conventions."""

from __future__ import annotations

import pytest

import run_all_tests as matrix


@pytest.mark.parametrize(
    ("profile", "expected_backend", "env_key"),
    [
        ("gemini", "gemini", "GEMINI_GENERATION_MODEL"),
        ("sonnet4.6", "anthropic", "ANTHROPIC_SONNET_MODEL"),
        ("gpt5.5", "openai", "OPENAI_GENERATION_MODEL"),
    ],
)
def test_resolve_gen_profile_direct_backends(
    matrix_runner_factory, profile, expected_backend, env_key
):
    runner = matrix_runner_factory()
    backend, model = runner.resolve_gen_profile(profile)
    assert backend == expected_backend
    assert model == runner.env[env_key]


def test_resolve_gen_profile_rejects_bedrock_gemini_pro(matrix_runner_factory):
    runner = matrix_runner_factory()
    with pytest.raises(ValueError, match="gemini-pro"):
        runner.resolve_gen_profile("gemini-pro")


def test_resolve_gen_profile_rejects_unknown_profile(matrix_runner_factory):
    runner = matrix_runner_factory()
    with pytest.raises(ValueError, match="Unknown generation profile"):
        runner.resolve_gen_profile("anthropic.claude-3-5-sonnet-20241022-v2:0")


def test_openai_profile_skipped_without_api_key(matrix_runner_factory):
    runner = matrix_runner_factory()
    runner.env.pop("OPENAI_API_KEY", None)
    calls: list[str] = []

    runner.ensure_csd_target_baselines = lambda *args: calls.append("baseline")
    runner.run_cmd = lambda *args, **kwargs: calls.append("run") or True

    assert runner.run_metadecode_case(
        "gsm_symbolic",
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "1",
        "30",
        "gpt5.5",
        "600",
    )
    assert calls == []


def test_openai_profile_available_with_api_key(matrix_runner_factory):
    runner = matrix_runner_factory()
    assert runner.openai_generation_available("gpt5.5") is True
    runner.env.pop("OPENAI_API_KEY", None)
    assert runner.openai_generation_available("gpt5.5") is False
    assert runner.openai_generation_available("gemini") is True


def test_slugify_normalizes_hf_model_ids():
    assert matrix.slugify("Qwen/Qwen2.5-Coder-7B-Instruct") == "Qwen_Qwen2.5_Coder_7B_Instruct"
    assert matrix.slugify("meta-llama/Llama-3.1-8B-Instruct") == "meta_llama_Llama_3.1_8B_Instruct"


def test_default_models_list_matches_matrix_runner():
    models = matrix.csv_list(matrix.DEFAULT_MODELS)
    assert "Qwen/Qwen2.5-Coder-7B-Instruct" in models
    assert len(models) == 4
