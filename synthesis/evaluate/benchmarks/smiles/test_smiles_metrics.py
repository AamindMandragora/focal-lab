"""Unit tests for SMILES scoring and grammar fallback."""

from __future__ import annotations

import unittest
from pathlib import Path

from synthesis.evaluate.benchmarks.smiles.grammar_helpers import (
    build_smiles_tier1_body_grammar,
    build_smiles_tier2_delimited_grammar,
)
from synthesis.evaluate.benchmarks.smiles.metrics import (
    evaluate_smiles_output,
    grammar_valid_with_fallback,
)


class SmilesGrammarFallbackTests(unittest.TestCase):
    def setUp(self) -> None:
        base = Path("synthesis/evaluate/grammars/smiles_acrylates.lark").read_text()
        self.base_grammar = base
        self.tier2_grammar = build_smiles_tier2_delimited_grammar(base)
        self.tier1_grammar = build_smiles_tier1_body_grammar(base)
        self.smiles = "CC(=C)C(=O)OCC1=CC=CC=C1C2=CC=CC=C2"
        self.exemplars = [
            "C=CC(=O)OCC1=CC=CC=C1",
            "C=CC(=O)OC1=CC=CC=C1",
        ]

    def test_tier2_body_fails_without_suffix(self) -> None:
        ok, source = grammar_valid_with_fallback(
            self.smiles,
            self.tier2_grammar,
            base_grammar_text=self.base_grammar,
        )
        self.assertTrue(ok)
        self.assertIn(source, {"base", "tier_closed"})

    def test_evaluate_smiles_output_syntax_valid_with_fallback(self) -> None:
        row = evaluate_smiles_output(
            "acrylates",
            self.smiles,
            self.tier2_grammar,
            self.exemplars,
            require_rdkit=False,
            base_grammar_text=self.base_grammar,
        )
        self.assertTrue(row["grammar_valid"])
        self.assertTrue(row["syntax_valid"])
        self.assertTrue(row["unique_valid_candidate"])


if __name__ == "__main__":
    unittest.main()
