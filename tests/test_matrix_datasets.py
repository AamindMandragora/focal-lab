"""Dataset registry, split manifests, and benchmark-key coverage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.conftest import REPO_ROOT, flag_value


@pytest.mark.parametrize(
    "dataset_name",
    ["gsm_symbolic", "spider", "smiles"],
)
def test_benchmark_registry_exposes_eval_logic(dataset_name):
    from synthesis.evaluate.benchmarks.registry import get_logic

    logic = get_logic(dataset_name)
    for attr in (
        "load_dataset_sample",
        "format_prompt",
        "expected_answer",
    ):
        assert hasattr(logic, attr), f"{dataset_name} logic missing {attr}"


def test_benchmark_registry_rejects_unknown_dataset():
    from synthesis.evaluate.benchmarks.registry import get_logic

    with pytest.raises(ValueError, match="Unknown dataset"):
        get_logic("mnist")


def test_tracked_split_manifests_exist_and_are_valid():
    gsm_path = REPO_ROOT / "environment" / "benchmark_splits" / "gsm_symbolic_crane_proportional.json"
    spider_path = REPO_ROOT / "environment" / "benchmark_splits" / "spider_dev_proportional.json"
    assert gsm_path.is_file()
    assert spider_path.is_file()

    gsm = json.loads(gsm_path.read_text())
    spider = json.loads(spider_path.read_text())
    assert isinstance(gsm.get("eval_indices"), list) and gsm["eval_indices"]
    assert isinstance(spider.get("test_indices"), list) and spider["test_indices"]


def test_smiles_dataset_loads_all_default_classes():
    from synthesis.evaluate.benchmarks.smiles.dataset import SMILES_CLASSES, load_smiles

    rows = load_smiles(classes=list(SMILES_CLASSES), samples_per_class=2)
    assert len(rows) == len(SMILES_CLASSES) * 2
    seen = {row["class_name"] for row in rows}
    assert seen == set(SMILES_CLASSES)
    for row in rows:
        assert row.get("grammar_path")
        assert row.get("prompt")


def test_smiles_registry_load_via_mock_evaluator():
    from synthesis.evaluate.benchmarks.registry import get_logic

    class _Eval:
        smiles_classes = ["acrylates"]
        sample_size = 3

    logic = get_logic("smiles")
    rows = logic.load_dataset_sample(_Eval())
    assert len(rows) == 3
    assert all(row["class_name"] == "acrylates" for row in rows)


def test_matrix_runner_sample_sizes_per_benchmark(matrix_runner_factory):
    runner = matrix_runner_factory()
    assert runner.generation_sample_size("gsm_symbolic") == "51"
    assert runner.generation_sample_size("spider") == "52"
    assert runner.generation_sample_size("smiles") == "52"
    assert runner.evaluation_sample_size("gsm_symbolic") == "50"
    assert runner.evaluation_sample_size("spider") == "100"
    assert runner.evaluation_sample_size("smiles") == "100"


def test_matrix_runner_smiles_class_variants(matrix_runner_factory):
    runner = matrix_runner_factory()
    assert runner.smiles_class_variants("smiles") == [
        "acrylates",
        "chain_extenders",
        "isocyanates",
    ]
    assert runner.smiles_class_variants("gsm_symbolic") == [""]
    assert runner.smiles_class_variants("spider") == [""]


def test_matrix_runner_benchmark_keys_and_baseline_paths(matrix_runner_factory):
    runner = matrix_runner_factory()
    assert runner.benchmark_key("smiles", "acrylates") == "smiles__class_acrylates"
    assert runner.benchmark_key("gsm", "") == "gsm_symbolic"

    cars_path = runner.fixed_baseline_path(
        "cars",
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "smiles",
        "1",
        "900",
        smiles_class="acrylates",
    )
    assert cars_path.name == "tb1__ms900__cs200.json"
    assert cars_path.parent.name == "cars"
    assert cars_path.parent.parent.name == "smiles__class_acrylates"

    rs_path = runner.fixed_baseline_path(
        "rs",
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "smiles",
        "1",
        "900",
        smiles_class="chain_extenders",
    )
    assert rs_path.name == "tb1__ms900__rs200.json"
    assert rs_path.parent.name == "rs"
    assert rs_path.parent.parent.name == "smiles__class_chain_extenders"


def test_gsm_split_name_falls_back_to_eval_when_train_empty(matrix_runner_factory):
    runner = matrix_runner_factory()
    assert runner.gsm_split_name_for_role("train") == "eval"
    assert runner.gsm_split_name_for_role("eval") == "eval"


def test_split_flags_on_generation_and_evaluation_commands(matrix_runner_factory):
    runner = matrix_runner_factory()
    gen_cmd: list[str] = []
    eval_cmd: list[str] = []
    runner.add_generation_split_flags(gen_cmd, "gsm_symbolic")
    runner.add_evaluation_split_flags(eval_cmd, "gsm_symbolic")
    assert flag_value(gen_cmd, "--gsm-split-name") == "eval"
    assert flag_value(eval_cmd, "--gsm-split-name") == "eval"

    gen_cmd = []
    eval_cmd = []
    runner.add_generation_split_flags(gen_cmd, "spider")
    runner.add_evaluation_split_flags(eval_cmd, "spider")
    assert flag_value(gen_cmd, "--spider-split-name") == "train"
    assert flag_value(eval_cmd, "--spider-split-name") == "eval"


def test_ensure_split_manifests_requires_tracked_files(matrix_runner_factory, tmp_path):
    runner = matrix_runner_factory()
    runner.config.gsm_split_file = str(tmp_path / "missing_gsm.json")
    with pytest.raises(SystemExit, match="GSM split manifest not found"):
        runner.ensure_split_manifests()


def test_eval_max_steps_per_benchmark(matrix_runner_factory):
    runner = matrix_runner_factory()
    assert runner.eval_max_steps_for("gsm_symbolic") == "900"
    assert runner.eval_max_steps_for("spider") == "600"
    assert runner.eval_max_steps_for("smiles") == "600"
