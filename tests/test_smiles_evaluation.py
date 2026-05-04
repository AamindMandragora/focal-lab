from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from lark import Lark

from evaluations.smiles.dataset import SMILES_CLASSES, DATA_DIR, get_smiles_task, load_smiles
from evaluations.smiles.metrics import evaluate_smiles_output, target_class_membership


def test_loads_all_cars_smiles_tasks():
    rows = load_smiles(samples_per_class=2)
    assert len(rows) == 2 * len(SMILES_CLASSES)
    assert {row["class_name"] for row in rows} == set(SMILES_CLASSES)
    for row in rows:
        assert row["prompt"].startswith("You are an expert in chemistry")
        assert row["grammar_text"].startswith("start: smiles")
        assert row["prompt_exemplars"]


def test_prompt_exemplars_are_not_unique_valid_candidates():
    task = get_smiles_task("acrylates")
    exemplar = task["prompt_exemplars"][0]
    result = evaluate_smiles_output(
        "acrylates",
        exemplar,
        task["grammar_text"],
        task["prompt_exemplars"],
    )
    assert result["class_membership"] is True
    assert result["is_prompt_exemplar"] is True
    assert result["unique_valid_candidate"] is False


def test_class_membership_patterns_match_required_motifs():
    assert target_class_membership("acrylates", "C=CC(=O)OCC")
    assert target_class_membership("chain_extenders", "OCCO")
    assert target_class_membership("chain_extenders", "N1CCNCC1")
    assert target_class_membership("isocyanates", "O=C=NCCCCCCN=C=O")
    assert not target_class_membership("isocyanates", "CCCCCC")


def test_grammar_parses_one_known_exemplar_per_class():
    for class_name in SMILES_CLASSES:
        task = get_smiles_task(class_name)
        parser = Lark(task["grammar_text"], start="start", parser="lalr")
        parser.parse(task["prompt_exemplars"][0])


def test_benchmark_script_dry_run_finds_cars_shape(tmp_path: Path):
    cars_repo = tmp_path / "cars"
    smiles_dir = cars_repo / "datasets" / "smiles"
    smiles_dir.mkdir(parents=True)
    (cars_repo / "run_task.py").write_text("print('placeholder')\n")
    for source in DATA_DIR.glob("acrylates.*"):
        shutil.copy(source, smiles_dir / source.name)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_smiles_vs_cars.py",
            "--dry-run",
            "--run-cars",
            "--cars-repo",
            str(cars_repo),
            "--classes",
            "acrylates",
            "--output-dir",
            str(tmp_path / "out"),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=True,
    )
    assert "Qwen/Qwen2.5-7B-Instruct" in result.stdout
    out_files = list((tmp_path / "out").glob("smiles_benchmark_*.json"))
    assert out_files
