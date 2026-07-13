import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_dry_run_does_not_require_configured_environment_to_exist():
    env = os.environ.copy()
    env["VAS_CONDA_ENV"] = "/path/that/does/not/exist"
    result = subprocess.run(
        [
            sys.executable,
            "run_all_tests.py",
            "--dry-run",
            "--skip-ablations",
            "--models",
            "Qwen/Qwen2.5-1.5B-Instruct",
            "--benchmarks",
            "gsm_symbolic",
            "--strategies",
            "unconstrained",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert "conda environment python not found" not in result.stderr


def test_tmux_launcher_and_docs_have_no_personal_environment_default():
    launcher = (REPO_ROOT / "run_tmux.sh").read_text()
    readme = (REPO_ROOT / "README.md").read_text()
    agents = (REPO_ROOT / "AGENTS.md").read_text()

    assert "/apps/conda/advayth2" not in launcher
    assert "/apps/conda/advayth2" not in readme
    assert "/apps/conda/advayth2" not in agents
    assert "METADECODE_TMUX_SESSION" in readme
    assert "VAS_CONDA_ENV" in launcher


def test_docs_match_executable_matrix_defaults_and_python_command():
    readme = (REPO_ROOT / "README.md").read_text()
    environment_readme = (REPO_ROOT / "environment/README.md").read_text()
    split_readme = (REPO_ROOT / "experiments/splits/README.md").read_text()
    legacy_readme = (REPO_ROOT / "legacy/README.md").read_text()

    assert "`Qwen/Qwen3.5-2B`** (first model" in readme
    assert "--accuracy-win-margin 0.0" in readme
    assert "python run_all_tests.py" not in readme
    assert "gsm_symbolic_crane_proportional_49x49_seed123.json" in environment_readme
    assert "gsm_symbolic_crane_proportional_49x49_seed123.json" in split_readme
    assert "environment/clone_legacy/repos.json" not in legacy_readme


def test_nested_docs_match_current_split_and_python3_contracts():
    split_docs = [
        REPO_ROOT / "environment/AGENTS.md",
        REPO_ROOT / "environment/benchmark_splits/AGENTS.md",
        REPO_ROOT / "experiments/README.md",
    ]
    for path in split_docs:
        text = path.read_text()
        assert "gsm_symbolic_crane_proportional_49x49_seed123.json" in text, path
        assert "spider_dev_proportional.json" in text, path

    command_docs = [
        REPO_ROOT / "run_tmux.sh",
        REPO_ROOT / "experiments/README.md",
        REPO_ROOT / "experiments/warmstarts/README.md",
        REPO_ROOT / "synthesis/scripts/README.md",
        REPO_ROOT / "synthesis/verify/reference/README.md",
    ]
    for path in command_docs:
        assert "python -m " not in path.read_text(), path


def test_claude_launch_guide_defers_to_current_agents_contract():
    guide = (REPO_ROOT / "CLAUDE.md").read_text()
    assert "CUDA_VISIBLE_DEVICES=2,3" not in guide
    assert '--generation-model "Qwen/Qwen2.5-Coder-7B-Instruct"' not in guide
    assert "AGENTS.md" in guide
