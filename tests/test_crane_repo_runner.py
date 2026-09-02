import argparse
import os
import sys

from synthesis.evaluate.baselines import crane_repo_runner
from synthesis.evaluate import run_legacy_fixed_strategy


def test_crane_child_uses_the_active_python_interpreter(monkeypatch, tmp_path):
    crane_repo = tmp_path / "CRANE"
    crane_src = crane_repo / "src"
    result_dir = crane_src / "logging" / "gsm_symbolic"
    result_dir.mkdir(parents=True)
    vendored_iter_syncode = crane_src / "itergen" / "iter_syncode"
    vendored_iter_syncode.mkdir(parents=True)
    (result_dir / "result.jsonl").write_text("{}\n")

    monkeypatch.setattr(crane_repo_runner, "CRANE_REPO_DIR", crane_repo)
    monkeypatch.setattr(crane_repo_runner, "CRANE_SRC_DIR", crane_src)
    monkeypatch.setattr(
        run_legacy_fixed_strategy,
        "_annotate_legacy_rows_with_syntax",
        lambda rows, args, dataset: rows,
    )
    monkeypatch.setattr(
        run_legacy_fixed_strategy,
        "_build_minimal_json",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        run_legacy_fixed_strategy,
        "_legacy_local_cuda_device",
        lambda device: "cuda:0",
    )

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs

    monkeypatch.setattr(crane_repo_runner.subprocess, "run", fake_run)

    args = argparse.Namespace(
        strategy="crane",
        eval_sample_size=1,
        eval_model="Qwen/Qwen3.5-2B",
        eval_max_steps=900,
        device="cuda",
        output_json=str(tmp_path / "baseline.json"),
        gsm_split_file=None,
    )

    assert crane_repo_runner.run_crane_repo_baseline(args, "gsm_symbolic") == 0
    assert captured["cmd"][0] == sys.executable
    assert captured["kwargs"]["check"] is True
    assert str(vendored_iter_syncode) in captured["kwargs"]["env"]["PYTHONPATH"].split(
        os.pathsep
    )
