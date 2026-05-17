import json
import os
import subprocess
import sys
from pathlib import Path


def _write_baseline_json(
    baseline_dir: Path,
    *,
    strategy: str,
    model_slug: str,
    benchmark_key: str,
    accuracy: float,
    syntax_rate: float,
) -> None:
    metrics = {}
    if strategy == "crane":
        metrics["adapter"] = "crane_shared_evaluator"
    path = (
        baseline_dir
        / strategy
        / model_slug
        / f"{benchmark_key}__tb1__ms900.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "accuracy": accuracy,
                "syntax_rate": syntax_rate,
                "metrics": metrics,
                "answers": [{"generated_answer": "SELECT 1"}],
            },
            indent=2,
        )
        + "\n"
    )


def _fake_conda_env(tmp_path: Path) -> dict[str, str]:
    fake_bin = tmp_path / "fake-bin"
    fake_base = tmp_path / "fake-conda-base"
    fake_env = tmp_path / "fake-conda-env"
    fake_bin.mkdir()
    (fake_base / "etc" / "profile.d").mkdir(parents=True)
    (fake_env / "bin").mkdir(parents=True)
    (fake_env / "lib").mkdir()

    conda_exe = fake_bin / "conda"
    conda_exe.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"$1\" == \"info\" && \"$2\" == \"--base\" ]]; then\n"
        f"  printf '%s\\n' {sh_quote(str(fake_base))}\n"
        "  exit 0\n"
        "fi\n"
        "exit 1\n"
    )
    conda_exe.chmod(0o755)

    (fake_base / "etc" / "profile.d" / "conda.sh").write_text(
        "conda() {\n"
        "  if [[ \"$1\" == \"activate\" ]]; then\n"
        "    export CONDA_PREFIX=\"$2\"\n"
        "    export PATH=\"$2/bin:$PATH\"\n"
        "    return 0\n"
        "  fi\n"
        "  return 1\n"
        "}\n"
    )

    python_wrapper = fake_env / "bin" / "python"
    python_wrapper.write_text(
        "#!/usr/bin/env bash\n"
        f"real_python={sh_quote(sys.executable)}\n"
        "if [[ \"$1\" == \"-\" ]]; then\n"
        "  script_file=$(mktemp)\n"
        "  cat > \"$script_file\"\n"
        "  if grep -q 'import rdkit' \"$script_file\"; then\n"
        "    rm -f \"$script_file\"\n"
        "    exit 0\n"
        "  fi\n"
        "  \"$real_python\" - \"${@:2}\" < \"$script_file\"\n"
        "  status=$?\n"
        "  rm -f \"$script_file\"\n"
        "  exit \"$status\"\n"
        "fi\n"
        "exec \"$real_python\" \"$@\"\n"
    )
    python_wrapper.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env.get('PATH', '')}"
    env["VAS_CONDA_ENV"] = str(fake_env)
    return env


def sh_quote(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"


def test_python_runner_dry_run_uses_strongest_cached_csd_baseline(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    baseline_dir = tmp_path / "baselines"
    generated_dir = tmp_path / "generated"
    model = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    model_slug = "Qwen_Qwen2.5_Coder_1.5B_Instruct"

    _write_baseline_json(
        baseline_dir,
        strategy="crane",
        model_slug=model_slug,
        benchmark_key="spider",
        accuracy=0.33,
        syntax_rate=0.92,
    )
    _write_baseline_json(
        baseline_dir,
        strategy="itergen",
        model_slug=model_slug,
        benchmark_key="spider",
        accuracy=0.16,
        syntax_rate=0.88,
    )
    _write_baseline_json(
        baseline_dir,
        strategy="cars",
        model_slug=model_slug,
        benchmark_key="spider",
        accuracy=0.66,
        syntax_rate=0.75,
    )

    env = _fake_conda_env(tmp_path)
    env["RUN_ALL_TESTS_CUDA_DEVICES"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "run_all_tests.py"),
            "--dry-run",
            "--skip-ablations",
            "--models",
            model,
            "--benchmarks",
            "spider",
            "--strategies",
            "metadecode",
            "--baseline-output-dir",
            str(baseline_dir),
            "--generated-output-dir",
            str(generated_dir),
        ],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=90,
    )

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "[target] metadecode spider" in output
    assert "best CSD baseline accuracy cars=66.0%, syntax crane=92.0%" in output
    assert "--min-accuracy 0.66" in output
    assert "--min-syntax-rate 0.92" in output


def test_python_runner_auto_selects_idle_gpu(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    env = _fake_conda_env(tmp_path)
    nvidia_smi = tmp_path / "fake-bin" / "nvidia-smi"
    nvidia_smi.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"$*\" == \"--query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits\" ]]; then\n"
        "  printf '0, 24000, 16000, 10\\n'\n"
        "  printf '1, 100, 39900, 0\\n'\n"
        "  exit 0\n"
        "fi\n"
        "exit 1\n"
    )
    nvidia_smi.chmod(0o755)

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "run_all_tests.py"),
            "--dry-run",
            "--skip-ablations",
            "--models",
            "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "--benchmarks",
            "spider",
            "--strategies",
            "unconstrained",
        ],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=90,
    )

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "[env] CUDA_VISIBLE_DEVICES=1" in output
    assert "[env] CUDA_VISIBLE_DEVICES=3" not in output


def test_python_runner_defaults_to_recompute_and_split_policy(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    baseline_dir = tmp_path / "baselines"
    generated_dir = tmp_path / "generated"
    split_dir = tmp_path / "splits"
    model = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

    env = _fake_conda_env(tmp_path)
    env["RUN_ALL_TESTS_CUDA_DEVICES"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "run_all_tests.py"),
            "--dry-run",
            "--skip-ablations",
            "--models",
            model,
            "--benchmarks",
            "gsm,spider,smiles",
            "--strategies",
            "metadecode",
            "--baseline-output-dir",
            str(baseline_dir),
            "--generated-output-dir",
            str(generated_dir),
            "--split-output-dir",
            str(split_dir),
            "--smiles-classes",
            "acrylates",
        ],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=90,
    )

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "baseline cache mode: refresh" in output
    assert "split policy: GSM generation=50/eval=50; other generation=50/eval=100" in output

    gsm_split = split_dir / "gsm_train50_eval50_seed123.json"
    spider_split = split_dir / "spider_train50_eval100_seed123.json"
    assert json.loads(gsm_split.read_text())["train_size"] == 50
    assert json.loads(gsm_split.read_text())["eval_size"] == 50
    assert json.loads(spider_split.read_text())["train_size"] == 50
    assert json.loads(spider_split.read_text())["eval_size"] == 100

    assert f"--gsm-split-file {gsm_split} --gsm-split-name eval" in output
    assert f"--gsm-split-file {gsm_split} --gsm-split-name train" in output
    assert f"--spider-split-file {spider_split} --spider-split-name eval" in output
    assert f"--spider-split-file {spider_split} --spider-split-name train" in output
    assert "--eval-sample-size 50" in output
    assert "--eval-sample-size 100" in output
    assert "--smiles-samples-per-class 50 --smiles-classes acrylates" in output
    assert "--sample-size 100 --max-steps 900 --step-token-budget 1" in output
