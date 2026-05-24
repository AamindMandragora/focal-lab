"""Tests for split_smiles_class_baselines."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from synthesis.scripts.split_smiles_class_baselines import (
    _answers_by_class,
    _needs_split,
    split_baselines,
)


def _answer(class_name: str, correct: bool = True) -> dict:
    return {
        "question": class_name,
        "prompt": f"prompt-{class_name}",
        "generated": "<<CCO>>",
        "extracted": "CCO",
        "correct": correct,
        "syntax_valid": True,
        "generation_seconds": 1.0,
    }


class SplitSmilesClassBaselinesTests(unittest.TestCase):
    def test_needs_split_multi_class(self) -> None:
        path = Path("smiles__class_acrylates__tb1__ms900.json")
        payload = {"answers": [_answer("acrylates"), _answer("chain_extenders")]}
        self.assertTrue(_needs_split(path, payload))

    def test_needs_split_ok_single_class(self) -> None:
        path = Path("smiles__class_acrylates__tb1__ms900.json")
        payload = {"answers": [_answer("acrylates")] * 100}
        self.assertFalse(_needs_split(path, payload))

    def test_split_writes_per_class_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "baselines" / "gcd" / "model"
            root.mkdir(parents=True)
            combined = [_answer(c) for c in ("acrylates", "chain_extenders", "isocyanates") for _ in range(2)]
            for cls in ("acrylates", "chain_extenders", "isocyanates"):
                path = root / f"smiles__class_{cls}__tb1__ms900.json"
                path.write_text(
                    json.dumps(
                        {
                            "accuracy": 0.5,
                            "syntax_rate": 1.0,
                            "metrics": {"num_examples": len(combined), "adapter": "gcd"},
                            "answers": combined,
                        }
                    )
                )
            n = split_baselines(
                baselines_root=Path(tmp) / "baselines",
                strategies=("gcd",),
                samples_per_class=2,
                dry_run=False,
                backup=False,
            )
            self.assertEqual(n, 3)
            acryl = json.loads((root / "smiles__class_acrylates__tb1__ms900.json").read_text())
            self.assertEqual(len(acryl["answers"]), 2)
            self.assertEqual(acryl["metrics"]["num_examples"], 2)
            self.assertTrue(acryl["metadata"]["smiles_class_split_from_combined"])
            self.assertEqual(acryl["accuracy"], 1.0)

    def test_answers_by_class(self) -> None:
        grouped = _answers_by_class([_answer("acrylates"), _answer("isocyanates")])
        self.assertEqual(len(grouped["acrylates"]), 1)
        self.assertEqual(len(grouped["isocyanates"]), 1)
        self.assertEqual(len(grouped["chain_extenders"]), 0)


if __name__ == "__main__":
    unittest.main()
