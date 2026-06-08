"""Metadecode target selection, synthesis launch, final eval, and retry queue."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import run_all_tests as matrix
from tests.conftest import REPO_ROOT, assert_flag_values, flag_value, write_baseline_json


def test_best_csd_baseline_targets_picks_maxima_and_clips_syntax(matrix_runner_factory):
    runner = matrix_runner_factory(dry_run=False)
    eval_model = "Qwen/Qwen2.5-Coder-7B-Instruct"
    for strategy, accuracy, syntax in [
        ("unconstrained", 0.30, 0.70),
        ("gcd", 0.35, 0.80),
        ("crane", 0.40, 0.95),
        ("itergen", 0.38, 0.99),
        ("rs", 0.32, 0.75),
    ]:
        path = runner.fixed_baseline_path(strategy, eval_model, "gsm_symbolic", "1", "900")
        adapter = "crane_legacy_crane" if strategy == "crane" else f"{strategy}_legacy"
        write_baseline_json(path, accuracy=accuracy, syntax_rate=syntax, adapter=adapter)

    (
        best_accuracy,
        acc_strategy,
        _acc_path,
        _acc_pct,
        best_syntax,
        syn_strategy,
        _syn_path,
        syn_pct,
    ) = runner.best_csd_baseline_targets("gsm_symbolic", eval_model, "1", "900")

    assert best_accuracy == pytest.approx(0.40)
    assert acc_strategy == "crane"
    assert best_syntax == pytest.approx(0.90)
    assert syn_strategy == "itergen"
    assert "clipped" in syn_pct


def test_accuracy_target_with_margin(matrix_runner_factory):
    runner = matrix_runner_factory()
    assert runner.accuracy_target_with_margin(0.42, "crane") == pytest.approx(0.45)
    assert runner.accuracy_target_with_margin(0.99, "crane") == pytest.approx(1.0)
    assert runner.accuracy_target_with_margin(0.50, "none") == 0.0


def test_ensure_csd_target_baselines_skips_when_targets_exist(matrix_runner_factory):
    runner = matrix_runner_factory(dry_run=False)
    eval_model = "Qwen/Qwen2.5-Coder-7B-Instruct"
    for strategy in matrix.BASELINE_TARGET_STRATEGIES:
        path = runner.fixed_baseline_path(strategy, eval_model, "gsm_symbolic", "1", "900")
        from synthesis.evaluate.baselines.registry import ADAPTER_IDS

        adapter = ADAPTER_IDS[strategy]
        write_baseline_json(path, accuracy=0.4, syntax_rate=0.9, adapter=adapter)

    calls: list[str] = []
    runner.run_fixed_strategy_case = lambda *args, **kwargs: calls.append("run") or True
    runner.ensure_csd_target_baselines("gsm_symbolic", eval_model, "1", "900")
    assert calls == []


@pytest.mark.parametrize(
    ("benchmark", "smiles_class", "max_steps", "sample_size", "expected_task", "dataset_flags"),
    [
        (
            "gsm_symbolic",
            "",
            "900",
            "51",
            "Solve math word problems step by step, wrapping intermediate symbolic expressions and the final answer inside << >> delimiters.",
            {"--gsm-split-file": "gsm_split.json", "--gsm-split-name": "eval"},
        ),
        (
            "spider",
            "",
            "600",
            "52",
            "Generate a single valid SQL query as exactly `SQL: <<YOUR QUERY>>`, using only the provided schema context.",
            {"--spider-split-file": "spider_split.json", "--spider-split-name": "train"},
        ),
        (
            "smiles",
            "acrylates",
            "600",
            "52",
            "Generate valid molecules in the requested class.",
            {"--smiles-classes": "acrylates"},
        ),
    ],
)
def test_metadecode_synthesis_command_contract(
    matrix_runner_factory,
    tmp_path,
    benchmark,
    smiles_class,
    max_steps,
    sample_size,
    expected_task,
    dataset_flags,
):
    runner = matrix_runner_factory()
    captured: list[list[str]] = []

    runner.ensure_csd_target_baselines = lambda *args: None
    runner.best_csd_baseline_targets = lambda *args: (
        0.42,
        "crane",
        "/tmp/crane.json",
        "42.0%",
        0.88,
        "itergen",
        "/tmp/itergen.json",
        "88.0%",
    )
    runner.run_cmd = lambda cmd, **kwargs: captured.append(cmd) or True

    assert runner.run_metadecode_case(
        benchmark,
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "1",
        "10",
        "sonnet4.6",
        max_steps,
        smiles_class,
    )

    assert len(captured) == 1
    cmd = captured[0]
    assert cmd[:3] == ["python", "-m", "synthesis.run_synthesis"]
    assert "--adaptive-helper-mask" in cmd
    assert "--temperature" not in cmd
    assert_flag_values(
        cmd,
        {
            "--task": expected_task,
            "--dataset": benchmark,
            "--generation-model": "claude-sonnet-4-6",
            "--generation-backend": "anthropic",
            "--eval-model": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "--max-iterations": "10",
            "--min-accuracy": "0.45",
            "--min-syntax-rate": "0.88",
            "--eval-sample-size": sample_size,
            "--eval-max-steps": max_steps,
            "--helper-selection-policy": "bandit",
            "--refinement-beam-size": "2",
        },
    )
    for flag, value in dataset_flags.items():
        actual = flag_value(cmd, flag)
        if flag.endswith("-file"):
            assert actual == str(tmp_path / value)
        else:
            assert actual == value


def test_metadecode_final_eval_command(matrix_runner_factory, tmp_path):
    runner = matrix_runner_factory()
    out_json = tmp_path / "metadecode_out.json"
    cmd = runner.metadecode_final_eval_command(
        tmp_path / "GeneratedCSD.py",
        out_json,
        "spider",
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "2",
        "600",
    )
    assert cmd[:4] == [
        "python",
        "-m",
        "synthesis.scripts.reevaluate_compiled_csd",
        str(tmp_path / "GeneratedCSD.py"),
    ]
    assert_flag_values(
        cmd,
        {
            "--dataset": "spider",
            "--eval-model": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "--sample-size": "100",
            "--max-steps": "600",
            "--step-token-budget": "2",
            "--spider-split-name": "eval",
        },
    )


def test_metadecode_success_path_runs_final_eval_and_annotates(
    matrix_runner_factory, tmp_path
):
    runner = matrix_runner_factory(dry_run=False)
    runner.ensure_csd_target_baselines = lambda *args: None
    runner.best_csd_baseline_targets = lambda *args: (
        0.42,
        "crane",
        "/tmp/crane.json",
        "42.0%",
        0.88,
        "itergen",
        "/tmp/itergen.json",
        "88.0%",
    )

    run_dir = tmp_path / "generated" / "run_001"
    compiled_dir = run_dir / "python" / "metadecode_test"
    compiled_dir.mkdir(parents=True)
    compiled_module = compiled_dir / "GeneratedCSD.py"
    compiled_module.write_text("# compiled\n")

    (run_dir / "results").mkdir()
    (run_dir / "results" / "success_report.json").write_text(
        json.dumps({"compiled_dir": str(compiled_dir)}) + "\n"
    )
    (tmp_path / "generated" / "latest_run.txt").write_text(str(run_dir) + "\n")

    commands: list[list[str]] = []

    def fake_run_cmd(cmd, **kwargs):
        commands.append(cmd)
        if "--output-json" in cmd:
            out_path = Path(flag_value(cmd, "--output-json"))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text('{"accuracy": 0.5, "syntax_rate": 0.9, "answers": []}\n')
        return True

    runner.run_cmd = fake_run_cmd

    assert runner.run_metadecode_case(
        "gsm_symbolic",
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "1",
        "10",
        "sonnet4.6",
        "900",
    )

    assert len(commands) == 2
    assert commands[0][:3] == ["python", "-m", "synthesis.run_synthesis"]
    assert commands[1][:3] == ["python", "-m", "synthesis.scripts.reevaluate_compiled_csd"]

    out_json = (
        runner.config.baseline_output_dir
        / "Qwen_Qwen2.5_Coder_7B_Instruct"
        / "gsm_symbolic"
        / "metadecode"
        / "gensonnet4.6__iter10__tb1__ms900.json"
    )
    payload = json.loads(out_json.read_text())
    assert payload["matrix_metadata"]["phase"] == "metadecode"
    assert payload["matrix_metadata"]["thresholds"]["min_accuracy"] == pytest.approx(0.45)


def test_metadecode_failure_on_non_gpu3_is_queued_for_retry(matrix_runner_factory):
    runner = matrix_runner_factory(dry_run=False)
    runner.env["CUDA_VISIBLE_DEVICES"] = "1"
    runner.ensure_csd_target_baselines = lambda *args: None
    runner.best_csd_baseline_targets = lambda *args: (
        0.42,
        "crane",
        "/tmp/crane.json",
        "42.0%",
        0.9,
        "itergen",
        "/tmp/itergen.json",
        "90.0%",
    )
    runner.run_cmd = lambda cmd, **kwargs: False

    assert runner.run_metadecode_case(
        "gsm_symbolic",
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "1",
        "40",
        "sonnet4.6",
        "600",
        phase="main_matrix",
    )

    queue_lines = runner.config.gpu3_retry_queue.read_text().splitlines()
    assert len(queue_lines) == 1
    record = json.loads(queue_lines[0])
    assert record["reason"] == "synthesis_subprocess_failed"
    assert record["case"]["phase"] == "main_matrix"
    assert record["cmd"][:3] == ["python", "-m", "synthesis.run_synthesis"]


def test_metadecode_failure_on_gpu3_is_not_requeued(matrix_runner_factory):
    runner = matrix_runner_factory(dry_run=False)
    runner.env["CUDA_VISIBLE_DEVICES"] = "3"
    runner.ensure_csd_target_baselines = lambda *args: None
    runner.best_csd_baseline_targets = lambda *args: (
        0.42,
        "crane",
        "/tmp/crane.json",
        "42.0%",
        0.9,
        "itergen",
        "/tmp/itergen.json",
        "90.0%",
    )
    runner.run_cmd = lambda cmd, **kwargs: False

    runner.run_metadecode_case(
        "gsm_symbolic",
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "1",
        "40",
        "sonnet4.6",
        "600",
        phase="main_matrix",
    )
    assert not runner.config.gpu3_retry_queue.exists()


def test_openai_quota_failure_does_not_abort_matrix(matrix_runner_factory):
    runner = matrix_runner_factory()
    runner.ensure_csd_target_baselines = lambda *args: None
    runner.best_csd_baseline_targets = lambda *args: (
        0.42,
        "crane",
        "/tmp/crane.json",
        "42.0%",
        0.9,
        "itergen",
        "/tmp/itergen.json",
        "90.0%",
    )
    calls: list[tuple] = []
    runner.run_cmd = lambda cmd, **kwargs: calls.append((cmd, kwargs)) or False

    runner.run_metadecode_case(
        "gsm_symbolic",
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "1",
        "30",
        "gpt5.5",
        "600",
    )

    assert len(calls) == 1
    assert calls[0][1]["abort_on_quota"] is False


def test_ablation_e_command_contract(matrix_runner_factory, tmp_path):
    runner = matrix_runner_factory()
    captured: list[tuple] = []

    runner.ensure_csd_target_baselines = lambda *args: None
    runner.best_csd_baseline_targets = lambda *args: (
        0.5,
        "crane",
        "/tmp/crane.json",
        "50.0%",
        0.9,
        "itergen",
        "/tmp/itergen.json",
        "90.0%",
    )
    runner.run_cmd = lambda cmd, **kwargs: captured.append((cmd, kwargs)) or True

    runner.run_ablation_e_case(
        "spider",
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "2",
        "--adaptive-helper-mask",
        "bandit",
    )

    assert len(captured) == 1
    cmd, run_kwargs = captured[0]
    assert run_kwargs["abort_on_quota"] is False
    assert cmd[:3] == ["python", "-m", "synthesis.run_synthesis"]
    assert "--adaptive-helper-mask" in cmd
    assert "--anthropic-thinking" not in cmd
    assert_flag_values(
        cmd,
        {
            "--dataset": "spider",
            "--generation-backend": "gemini",
            "--generation-model": "gemini-3-pro-preview",
            "--max-iterations": "30",
            "--min-accuracy": "0.53",
            "--helper-selection-policy": "bandit",
            "--refinement-beam-size": "2",
            "--spider-split-file": str(tmp_path / "spider_split.json"),
        },
    )


def test_matrix_result_json_annotation_records_provenance(matrix_runner_factory, tmp_path):
    runner = matrix_runner_factory(dry_run=False)
    result_json = tmp_path / "result.json"
    result_json.write_text('{"accuracy": 0.5, "syntax_rate": 0.9, "answers": []}\n')

    runner.annotate_result_json(
        result_json,
        runner.matrix_case_metadata(
            phase="ablation_synthesizer_model",
            strategy="metadecode",
            benchmark="spider",
            eval_model="Qwen/Qwen2.5-Coder-7B-Instruct",
            token_budget="1",
            max_steps="600",
            command=["python", "-m", "synthesis.run_synthesis"],
            synth_iter="10",
            gen_profile="gemini",
            generation_backend="gemini",
            generation_model="gemini-3-pro-preview",
            required_accuracy=0.53,
            required_syntax=0.9,
            target_accuracy_strategy="crane",
            target_accuracy_path="/tmp/crane.json",
            target_syntax_strategy="itergen",
            target_syntax_path="/tmp/itergen.json",
        ),
    )

    payload = json.loads(result_json.read_text())
    metadata = payload["matrix_metadata"]
    assert metadata["phase"] == "ablation_synthesizer_model"
    assert metadata["synthesis_controls"]["helper_selection_policy"] == "bandit"
    assert metadata["synthesis_controls"]["refinement_beam_size"] == 2
    assert metadata["thresholds"]["min_accuracy"] == 0.53
    assert metadata["splits"]["spider"]["split_name"] == "eval"


def test_run_synthesis_help_advertises_matrix_defaults():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "synthesis.run_synthesis", "--help"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    help_text = result.stdout
    assert "bandit" in help_text
    assert "--refinement-beam-size" in help_text
    assert "eval-max-seconds-per-example" in help_text


def test_metadecode_tasks_use_visible_delimiters():
    text = (REPO_ROOT / "run_all_tests.py").read_text()
    assert "SQL: <<YOUR QUERY>>" in text
    assert "You may optionally reason" not in text
