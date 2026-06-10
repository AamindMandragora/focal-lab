from synthesis.evaluate.baseline_store import build_minimal_baseline_from_rows
from synthesis.evaluate.benchmarks.smiles.pooled_eval import (
    DEFAULT_SMILES_POOLED_SUCCESS_TARGET,
)


def test_build_minimal_baseline_uses_unique_over_success_target_for_smiles():
    target = DEFAULT_SMILES_POOLED_SUCCESS_TARGET
    rows = [
        {
            "class_name": "acrylates",
            "question": "acrylates",
            "prompt": "prompt",
            "generated": "<<CCO>>",
            "extracted": "CCO",
            "correct": True,
            "syntax_valid": True,
            "smiles_eval": {
                "smiles": "CCO",
                "syntax_valid": True,
                "rdkit_available": True,
                "rdkit_valid": True,
                "class_membership": True,
                "valid_class_membership": True,
            },
        },
        {
            "class_name": "acrylates",
            "question": "acrylates",
            "prompt": "prompt",
            "generated": "<<CCO>>",
            "extracted": "CCO",
            "correct": False,
            "syntax_valid": True,
            "smiles_eval": {
                "smiles": "CCO",
                "syntax_valid": True,
                "rdkit_available": True,
                "rdkit_valid": True,
                "class_membership": True,
                "valid_class_membership": True,
            },
        },
        {
            "class_name": "acrylates",
            "question": "acrylates",
            "prompt": "prompt",
            "generated": "<<CCC>>",
            "extracted": "CCC",
            "correct": True,
            "syntax_valid": True,
            "smiles_eval": {
                "smiles": "CCC",
                "syntax_valid": True,
                "rdkit_available": True,
                "rdkit_valid": True,
                "class_membership": False,
                "valid_class_membership": False,
            },
        },
    ]
    payload = build_minimal_baseline_from_rows(
        rows,
        dataset="smiles",
        metadata={"success_target": target},
    )
    assert payload["accuracy"] == 1 / target
    assert payload["syntax_rate"] == 2 / target
    assert payload["accuracy_definition"] == "unique_in_class_over_success_target"
    assert payload["metrics"]["unique_syntax_valid_count"] == 2
    assert payload["metrics"]["unique_in_class_count"] == 1
