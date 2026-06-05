"""Matrix scheduling across benchmarks, models, strategies, and ablations."""

from __future__ import annotations

import pytest

import run_all_tests as matrix


def test_default_matrix_constants():
    assert matrix.DEFAULT_BENCHMARKS == "gsm,spider,smiles"
    assert matrix.DEFAULT_MAIN_SYNTH_ITERS == "40"
    assert matrix.DEFAULT_MAIN_GEN_PROFILE == "gemini"
    assert matrix.csv_list(matrix.DEFAULT_GEN_MODELS) == ["sonnet4.6", "gpt5.5"]
    assert "900" in matrix.csv_list(matrix.DEFAULT_STEP_BUDGETS)
    assert "cars" not in matrix.BASELINE_TARGET_STRATEGIES


def test_main_matrix_schedules_all_benchmarks_and_strategies(matrix_runner_factory):
    runner = matrix_runner_factory()
    runner.config.models = ["Qwen/Qwen2.5-1.5B-Instruct"]
    runner.config.benchmarks = matrix.csv_list(matrix.DEFAULT_BENCHMARKS)
    runner.config.strategies = matrix.csv_list(matrix.DEFAULT_STRATEGIES)
    runner.config.token_budgets = ["1"]
    runner.config.synth_iters = ["3"]
    runner.config.main_gen_profile = "gemini"
    seen: list[tuple[str, ...]] = []

    runner.run_fixed_strategy_cases = lambda strategy, benchmark, *args, **kwargs: seen.append(
        ("fixed", strategy, benchmark)
    )
    runner.run_metadecode_cases = lambda benchmark, *args, **kwargs: seen.append(
        ("metadecode", benchmark)
    )

    runner.run_main_matrix()

    benchmarks = {entry[-1] for entry in seen}
    assert benchmarks == {"gsm_symbolic", "spider", "smiles"}
    assert ("metadecode", "gsm_symbolic") in seen
    assert ("metadecode", "spider") in seen
    assert ("metadecode", "smiles") in seen
    for strategy in ("unconstrained", "gcd", "crane", "itergen", "rs"):
        assert any(entry[1] == strategy for entry in seen)


def test_main_metadecode_uses_40_iterations_and_gemini(matrix_runner_factory):
    runner = matrix_runner_factory()
    runner.config.synth_iters = matrix.csv_list(matrix.DEFAULT_SYNTH_ITERS)
    runner.config.models = ["Qwen/Qwen2.5-1.5B-Instruct"]
    runner.config.benchmarks = ["gsm"]
    runner.config.strategies = ["metadecode"]
    runner.config.token_budgets = ["1"]
    runner.config.main_gen_profile = "gemini"
    calls: list[tuple] = []
    runner.run_metadecode_cases = lambda *args, **kwargs: calls.append((args, kwargs))

    runner.run_main_matrix()

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[3] == "40"
    assert args[4] == "gemini"
    assert args[5] == "900"
    assert kwargs["phase"] == "main_matrix"


def test_ablation_sections_cli_normalizes_and_rejects_unknown():
    assert matrix.normalize_ablation_sections("c,e") == {"C", "E"}
    assert matrix.normalize_ablation_sections(" A, b ") == {"A", "B"}

    with pytest.raises(SystemExit, match="Invalid ablation section"):
        matrix.normalize_ablation_sections("A,Z")

    with pytest.raises(SystemExit, match="At least one ablation section"):
        matrix.normalize_ablation_sections("")


def test_ablation_c_runs_sonnet_and_gpt_profiles(matrix_runner_factory):
    runner = matrix_runner_factory()
    runner.config.benchmarks = ["gsm"]
    runner.config.step_budgets = []
    runner.config.token_budgets = ["1"]
    runner.config.synth_iters = ["10"]
    runner.config.gen_models = ["sonnet4.6", "gpt5.5"]
    runner.config.ablation_sections = {"C"}
    calls: list[tuple] = []

    runner.run_fixed_strategy_cases = lambda *args, **kwargs: None
    runner.run_ablation_e_case = lambda *args, **kwargs: None
    runner.run_metadecode_cases = lambda *args, **kwargs: calls.append(args)

    runner.run_ablations()

    assert sorted(call[4] for call in calls) == ["gpt5.5", "sonnet4.6"]


def test_ablation_section_filter_limits_scheduled_work(matrix_runner_factory):
    runner = matrix_runner_factory()
    runner.config.benchmarks = ["gsm"]
    runner.config.step_budgets = ["256"]
    runner.config.token_budgets = ["1", "2"]
    runner.config.synth_iters = ["3", "30"]
    runner.config.gen_models = ["sonnet4.6", "gpt5.5"]
    runner.config.ablation_sections = {"C"}
    calls: list[tuple[str, dict]] = []

    runner.run_fixed_strategy_cases = lambda *args, **kwargs: calls.append(("fixed", kwargs))
    runner.run_ablation_e_case = lambda *args, **kwargs: calls.append(("helper_mask", {}))
    runner.run_metadecode_cases = lambda *args, **kwargs: calls.append(("metadecode", kwargs))

    runner.run_ablations()

    assert all(kind == "metadecode" for kind, _kwargs in calls)
    assert [kwargs["phase"] for _kind, kwargs in calls] == [
        "ablation_synthesizer_model",
        "ablation_synthesizer_model",
    ]


def test_ablation_e_iterates_smiles_classes(matrix_runner_factory):
    runner = matrix_runner_factory()
    runner.config.benchmarks = ["smiles"]
    runner.config.ablation_sections = {"E"}
    runner.config.step_budgets = []
    runner.config.token_budgets = ["1"]
    calls: list[str] = []

    runner.run_fixed_strategy_cases = lambda *args, **kwargs: None
    runner.run_metadecode_cases = lambda *args, **kwargs: None
    runner.run_ablation_e_case = (
        lambda benchmark, eval_model, beam_size, mask_flag, policy, smiles_class="": calls.append(
            smiles_class or "<none>"
        )
    )

    runner.run_ablations()

    assert set(calls) == {"acrylates", "chain_extenders", "isocyanates"}
    assert len(calls) == 6  # 3 classes × 2 mask flags


def test_smiles_fixed_strategy_cases_expand_per_class(matrix_runner_factory):
    runner = matrix_runner_factory()
    runner.config.strategies = ["cars"]
    runner.config.benchmarks = ["smiles"]
    runner.config.models = ["Qwen/Qwen2.5-Coder-7B-Instruct"]
    runner.config.token_budgets = ["1"]
    classes: list[str] = []

    def capture(strategy, benchmark, eval_model, token_budget, max_steps, phase="baseline"):
        for smiles_class in runner.smiles_class_variants(benchmark):
            classes.append(smiles_class)

    runner.run_fixed_strategy_cases = capture
    runner.run_metadecode_cases = lambda *args, **kwargs: None
    runner.run_main_matrix()

    assert classes == ["acrylates", "chain_extenders", "isocyanates"]
