import hashlib
from pathlib import Path
from types import SimpleNamespace

from synthesis.scripts.reevaluate_compiled_csd import build_reevaluation_provenance


def test_reevaluation_provenance_binds_output_to_strategy_model_and_cell(tmp_path):
    csd = tmp_path / "GeneratedCSD.py"
    csd.write_text("# frozen strategy\n", encoding="utf-8")
    args = SimpleNamespace(
        dataset="smiles",
        eval_model="Qwen/Qwen3.5-9B",
        sample_size=100,
        max_steps=400,
        step_token_budget=1,
        smiles_classes="isocyanates",
        provenance_cell_id="smiles-qwen35-9b-isocyanates",
        provenance_manifest_commit="a" * 40,
    )

    provenance = build_reevaluation_provenance(args, csd)

    assert provenance == {
        "cell_id": "smiles-qwen35-9b-isocyanates",
        "manifest_commit": "a" * 40,
        "dataset": "smiles",
        "eval_model": "Qwen/Qwen3.5-9B",
        "compiled_csd_path": str(csd.resolve()),
        "compiled_csd_sha256": hashlib.sha256(csd.read_bytes()).hexdigest(),
        "sample_size": 100,
        "max_steps": 400,
        "step_token_budget": 1,
        "smiles_class": "isocyanates",
    }
