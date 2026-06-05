"""Shared fixtures for the matrix / metadecode test suite."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from synthesis.evaluate.feedback_loop import SynthesisPipeline


REPO_ROOT = Path(__file__).resolve().parents[1]


def flag_value(cmd: list[str], flag: str) -> str:
    assert flag in cmd, f"missing {flag} in {cmd}"
    index = cmd.index(flag)
    assert index + 1 < len(cmd), f"{flag} has no value in {cmd}"
    return cmd[index + 1]


def assert_flag_values(cmd: list[str], expected: dict[str, object]) -> None:
    for flag, value in expected.items():
        assert flag_value(cmd, flag) == str(value)


def dummy_pipeline(**kwargs) -> SynthesisPipeline:
    return SynthesisPipeline(
        evaluator=object(),
        generator=object(),
        verifier=object(),
        compiler=object(),
        **kwargs,
    )


def write_baseline_json(
    path: Path,
    *,
    accuracy: float,
    syntax_rate: float,
    adapter: str = "crane_legacy_crane",
    rows: int = 3,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    answers = []
    for i in range(rows):
        answers.append(
            {
                "question": f"q{i}",
                "prompt": f"p{i}",
                "generated": f"gen{i}",
                "extracted": f"ans{i}",
                "generated_answer": f"ans{i}",
                "gold_answer": f"ans{i}",
                "correct": True,
                "syntax_valid": True,
            }
        )
    payload = {
        "accuracy": accuracy,
        "syntax_rate": syntax_rate,
        "answers": answers,
        "metrics": {"adapter": adapter},
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


@pytest.fixture
def matrix_runner_factory(tmp_path):
    """Build a ``run_all_tests.Runner`` with temp output dirs and split manifests."""

    def _factory(*, dry_run: bool = True, **config_overrides):
        import run_all_tests as matrix

        gsm_split = tmp_path / "gsm_split.json"
        gsm_split.write_text(
            json.dumps({"train_indices": [], "eval_indices": [0, 1, 2]}) + "\n"
        )
        spider_split = tmp_path / "spider_split.json"
        spider_split.write_text(
            json.dumps({"train_indices": [0], "test_indices": [1, 2, 3]}) + "\n"
        )

        defaults = dict(
            models=["Qwen/Qwen2.5-Coder-7B-Instruct"],
            benchmarks=matrix.csv_list(matrix.DEFAULT_BENCHMARKS),
            strategies=["metadecode"],
            token_budgets=["1", "2", "4"],
            synth_iters=["3", "5", "10", "30"],
            gen_models=["sonnet4.6", "gpt5.5"],
            main_gen_profile="gemini",
            step_budgets=["256", "512", "900", "1024"],
            ablation_sections=set(matrix.VALID_ABLATION_SECTIONS),
            eval_backend="vllm",
            device="auto",
            generation_sample_size="52",
            eval_sample_size="100",
            gsm_generation_sample_size="51",
            gsm_eval_sample_size="50",
            eval_max_steps="600",
            eval_max_steps_gsm="900",
            rs_search_steps="200",
            cars_search_steps="200",
            smiles_classes=["acrylates", "chain_extenders", "isocyanates"],
            smiles_samples_per_class="100",
            eval_max_seconds_per_example="90",
            eval_min_examples_before_threshold_stop="15",
            accuracy_win_margin=0.03,
            synthesis_max_tokens="32768",
            restart_after_stuck_iters="0",
            helper_selection_policy="bandit",
            refinement_beam_size="2",
            anthropic_thinking="adaptive",
            anthropic_effort="xhigh",
            anthropic_thinking_display="summarized",
            vllm_gpu_memory_utilization="0.80",
            vllm_tensor_parallel_size=1,
            dafny_path="/tmp/dafny",
            generated_output_dir=tmp_path / "generated",
            baseline_output_dir=tmp_path / "baselines",
            ablation_output_dir=tmp_path / "ablations",
            baseline_cache_mode="reuse",
            gsm_split_file=str(gsm_split),
            spider_split_file=str(spider_split),
            dry_run=dry_run,
            skip_main=False,
            skip_ablations=False,
            conda_env_path=tmp_path / "conda",
            cuda_devices="0",
            cuda_oom_fallback="",
            free_gpu_max_used_mb=1024,
            gpu_wait_seconds=1,
            gpu_wait_timeout_seconds=0,
            main_synthesis_iterations="40",
            gpu3_retry_queue=tmp_path / "gpu3_retry_queue.jsonl",
            gpu3_retry_enabled=True,
        )
        defaults.update(config_overrides)
        config = matrix.Config(**defaults)
        return matrix.Runner(
            config=config,
            env={
                "ANTHROPIC_SONNET_MODEL": "claude-sonnet-4-6",
                "OPENAI_API_KEY": "test-openai-key",
                "OPENAI_GENERATION_MODEL": "gpt-5.5",
                "GEMINI_GENERATION_MODEL": "gemini-3-pro-preview",
            },
        )

    return _factory
