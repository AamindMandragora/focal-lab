"""Fixed-strategy baseline registry and command construction."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import assert_flag_values, flag_value, write_baseline_json


@pytest.mark.parametrize("strategy", ["unconstrained", "gcd", "crane", "itergen", "cars", "rs"])
def test_baseline_registry_lists_all_strategies(strategy):
    from synthesis.evaluate.baselines.registry import ADAPTER_IDS, STRATEGIES

    assert strategy in STRATEGIES
    assert strategy in ADAPTER_IDS


def test_baseline_registry_routes_smiles_through_smiles_module():
    from synthesis.evaluate.baselines.registry import run_baseline_strategy

    args = MagicMock(dataset="smiles", strategy="crane", output_json="/tmp/out.json")
    with patch("synthesis.evaluate.baselines.smiles.run", return_value=0) as smiles_run:
        assert run_baseline_strategy(args) == 0
        smiles_run.assert_called_once_with(args)


@pytest.mark.parametrize(
    "strategy",
    ["unconstrained", "gcd", "crane", "itergen", "cars", "rs"],
)
def test_baseline_registry_routes_non_smiles_to_strategy_adapter(strategy):
    from synthesis.evaluate.baselines.registry import run_baseline_strategy

    args = MagicMock(dataset="gsm", strategy=strategy, output_json="/tmp/out.json")
    module_name = f"synthesis.evaluate.baselines.{strategy}"
    with patch(f"{module_name}.run", return_value=0) as adapter_run:
        assert run_baseline_strategy(args) == 0
        adapter_run.assert_called_once_with(args)


def test_baseline_registry_rejects_unknown_strategy():
    from synthesis.evaluate.baselines.registry import run_baseline_strategy

    args = MagicMock(dataset="gsm", strategy="unknown", output_json="/tmp/out.json")
    with pytest.raises(ValueError, match="Unknown baseline strategy"):
        run_baseline_strategy(args)


@pytest.mark.parametrize(
    ("strategy", "benchmark", "smiles_class", "extra_flags"),
    [
        ("unconstrained", "gsm_symbolic", "", {}),
        ("gcd", "spider", "", {}),
        ("crane", "gsm_symbolic", "", {}),
        ("itergen", "spider", "", {}),
        ("cars", "smiles", "acrylates", {"--cars-search-steps": "200", "--smiles-classes": "acrylates"}),
        ("rs", "smiles", "chain_extenders", {"--rs-search-steps": "200", "--smiles-classes": "chain_extenders"}),
    ],
)
def test_fixed_strategy_command_includes_dataset_and_strategy_flags(
    matrix_runner_factory,
    strategy,
    benchmark,
    smiles_class,
    extra_flags,
):
    runner = matrix_runner_factory()
    captured: list[list[str]] = []
    runner.run_cmd = lambda cmd, **kwargs: captured.append(cmd) or True

    assert runner.run_fixed_strategy_case(
        strategy,
        benchmark,
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "1",
        runner.eval_max_steps_for(benchmark),
        smiles_class,
    )

    assert len(captured) == 1
    cmd = captured[0]
    assert cmd[:5] == [
        "python",
        "-m",
        "synthesis.evaluate.run_legacy_fixed_strategy",
        "--strategy",
        strategy,
    ]
    assert_flag_values(
        cmd,
        {
            "--dataset": benchmark,
            "--eval-model": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "--eval-backend": "vllm",
            "--eval-step-token-budget": "1",
        },
    )
    for flag, value in extra_flags.items():
        assert flag_value(cmd, flag) == value


def test_fixed_strategy_reuses_complete_cached_baseline(matrix_runner_factory, tmp_path):
    runner = matrix_runner_factory(dry_run=False)
    out_json = runner.fixed_baseline_path(
        "crane",
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "gsm_symbolic",
        "1",
        "900",
    )
    write_baseline_json(out_json, accuracy=0.4, syntax_rate=0.9, adapter="crane_legacy_crane")
    calls: list[list[str]] = []
    runner.run_cmd = lambda cmd, **kwargs: calls.append(cmd) or True

    assert runner.run_fixed_strategy_case(
        "crane",
        "gsm_symbolic",
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "1",
        "900",
    )
    assert calls == []


def test_baseline_json_matches_strategy_accepts_legacy_crane_adapters(
    matrix_runner_factory, tmp_path
):
    runner = matrix_runner_factory()
    from synthesis.evaluate.baselines.registry import CRANE_ADAPTER_IDS

    for adapter in CRANE_ADAPTER_IDS:
        path = tmp_path / f"{adapter}.json"
        write_baseline_json(path, accuracy=0.5, syntax_rate=0.9, adapter=adapter)
        assert runner.baseline_json_matches_strategy(path, "crane") is True

    bad = tmp_path / "bad.json"
    write_baseline_json(bad, accuracy=0.5, syntax_rate=0.9, adapter="unknown")
    assert runner.baseline_json_matches_strategy(bad, "crane") is False


def test_baseline_case_key_includes_search_steps(matrix_runner_factory):
    runner = matrix_runner_factory()
    model_slug = "Qwen_Qwen2.5_Coder_7B_Instruct"
    rs_key = runner.baseline_case_key("rs", model_slug, "gsm_symbolic", "1", "900")
    assert rs_key == ("rs", model_slug, "gsm_symbolic", "1", "900", "200")
    cars_key = runner.baseline_case_key(
        "cars", model_slug, "smiles__class_acrylates", "1", "900"
    )
    assert cars_key[-1] == "200"
