import json
import os
import re
import subprocess
from pathlib import Path

from run_all_tests import DEFAULT_GSM_SPLIT_FILE


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_default_gsm_matrix_split_is_nonempty_and_disjoint():
    runner = (REPO_ROOT / "run_all_tests.py").read_text()
    manifest = json.loads(DEFAULT_GSM_SPLIT_FILE.read_text())
    train = set(manifest.get("train_indices", []))
    evaluation = set(manifest.get("eval_indices", []))
    assert train
    assert evaluation
    assert train.isdisjoint(evaluation)
    assert "if role == \"train\" and not manifest.get(\"train_indices\")" not in runner


def test_legacy_repositories_are_pinned_to_commits():
    repos = json.loads((REPO_ROOT / "environment/legacy/repos.json").read_text())
    for name, config in repos.items():
        commit = config.get("commit", "")
        assert re.fullmatch(r"[0-9a-f]{40}", commit), f"{name} is not pinned"
    clone_script = (REPO_ROOT / "environment/clone_legacy_csds.sh").read_text()
    assert "environment/legacy/repos.json" in clone_script


def test_tmux_launcher_leaves_cuda_selection_to_caller_or_runner():
    launcher = (REPO_ROOT / "run_tmux.sh").read_text()
    assert "export CUDA_VISIBLE_DEVICES=" not in launcher


def test_readme_names_recorded_gsm_and_spider_reproduction_splits():
    readme = (REPO_ROOT / "README.md").read_text()
    assert "gsm_symbolic_crane_proportional_49x49_seed123.json" in readme
    assert "spider_dev_proportional_300x300_seed334.json" in readme


def _make_local_git_repo(path: Path) -> str:
    path.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)
    (path / "README.md").write_text("fixture\n")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=path, check=True)
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()


def test_legacy_clone_setup_is_repeatable(tmp_path):
    sources = {}
    for name in ("CRANE", "itergen", "cars"):
        path = tmp_path / f"source-{name}"
        sources[name] = (path, _make_local_git_repo(path))

    env = os.environ.copy()
    env.update({
        "LEGACY_ROOT": str(tmp_path / "clones"),
        "LEGACY_PATCH_ROOT": str(tmp_path / "empty-patches"),
        "LEGACY_SHALLOW": "0",
    })
    (tmp_path / "empty-patches").mkdir()
    for name, env_name in (("CRANE", "CRANE"), ("itergen", "ITERGEN"), ("cars", "CARS")):
        path, commit = sources[name]
        env[f"LEGACY_{env_name}_URL"] = str(path)
        env[f"LEGACY_{env_name}_REF"] = commit

    script = REPO_ROOT / "environment/clone_legacy_csds.sh"
    first = subprocess.run(["bash", str(script)], env=env, text=True, capture_output=True)
    second = subprocess.run(["bash", str(script)], env=env, text=True, capture_output=True)
    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr

    for name, (_, commit) in sources.items():
        clone = tmp_path / "clones" / name
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=clone, text=True).strip()
        assert actual == commit
    assert not list((tmp_path / "clones").rglob("*.rej"))
    assert not list((tmp_path / "clones").rglob("*.orig"))
