from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_RUNTIME_PACKAGES = {
    "google-auth",
    "interegular",
    "lark",
    "fire",
    "matplotlib",
    "nltk",
    "numpy",
    "python-dotenv",
    "rdkit",
    "regex",
    "tqdm",
    "z3-solver",
}


def _declared_packages(manifest: str) -> set[str]:
    declared = set()
    for raw_line in (REPO_ROOT / manifest).read_text().splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        name = line.split("@", 1)[0].split("[", 1)[0]
        for separator in ("==", ">=", "<=", "~=", "!=", ">", "<"):
            name = name.split(separator, 1)[0]
        declared.add(name.strip().lower())
    return declared


def test_root_manifest_declares_direct_runtime_dependencies():
    missing = REQUIRED_RUNTIME_PACKAGES - _declared_packages("requirements.txt")
    assert not missing, f"requirements.txt is missing: {sorted(missing)}"


def test_mac_manifest_declares_direct_runtime_dependencies():
    missing = REQUIRED_RUNTIME_PACKAGES - _declared_packages("requirements-mac.txt")
    assert not missing, f"requirements-mac.txt is missing: {sorted(missing)}"


def test_runtime_output_directories_are_ignored():
    rules = {
        line.strip().lstrip("/")
        for line in (REPO_ROOT / ".gitignore").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert "cache/" in rules
    assert "outputs/" in rules


def test_spider_parser_does_not_download_nltk_data_at_import_time():
    source = (
        REPO_ROOT
        / "synthesis/evaluate/syncode/syncode/utils/sql_spider_eval/process_sql.py"
    ).read_text()
    assert "nltk.download(" not in source


def test_mxeval_installer_pins_and_verifies_a_commit():
    source = (REPO_ROOT / "environment/install_mxeval_into_env.sh").read_text()
    assert "MXEVAL_COMMIT=" in source
    assert "rev-parse HEAD" in source
    assert "PYTHON_BIN=" in source
