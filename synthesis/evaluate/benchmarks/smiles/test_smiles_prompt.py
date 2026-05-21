"""Unit tests for SMILES prompt rendering and tier grammars."""

from __future__ import annotations

import unittest

from synthesis.evaluate.benchmarks.smiles.dataset import (
    SMILES_FEWSHOT_COUNT,
    get_smiles_task,
    prompt_exemplars_for_class,
)
from synthesis.evaluate.benchmarks.smiles.grammar_helpers import (
    build_smiles_tier1_body_grammar,
    build_smiles_tier2_delimited_grammar,
)
from synthesis.evaluate.prompt_tiers import (
    SMILES_TIER1_MAX_NEW_TOKENS,
    SMILES_TIER2_MAX_NEW_TOKENS,
    benchmark_max_new_tokens,
    effective_max_new_tokens,
    render_smiles_cars_prompt,
    smiles_tier1_max_new_tokens,
    smiles_tier2_max_new_tokens,
)


class SmilesPromptTests(unittest.TestCase):
    def test_frozen_exemplar_count_is_eight(self) -> None:
        for class_name in ("acrylates", "chain_extenders", "isocyanates"):
            exemplars = prompt_exemplars_for_class(class_name)
            self.assertEqual(len(exemplars), SMILES_FEWSHOT_COUNT)

    def test_tier1_prompt_has_no_trailing_molecule_slot(self) -> None:
        task = get_smiles_task("acrylates")
        prompt = render_smiles_cars_prompt(task, tier=1)
        self.assertIn("exactly 8", prompt.lower())
        self.assertIn("Molecule: C=CC", prompt)
        self.assertFalse(prompt.rstrip().endswith("Molecule:"))
        self.assertNotIn("Reasoning:", prompt)

    def test_tier2_prompt_ends_with_reasoning_and_delimiter_hint(self) -> None:
        task = get_smiles_task("chain_extenders")
        prompt = render_smiles_cars_prompt(task, tier=2)
        self.assertIn("<<", prompt)
        self.assertTrue(prompt.rstrip().endswith("Reasoning:"))
        self.assertFalse(prompt.rstrip().endswith("Molecule:"))

    def test_tier2_grammar_allows_closing_gt_gt(self) -> None:
        task = get_smiles_task("isocyanates")
        grammar = build_smiles_tier2_delimited_grammar(task["grammar_text"])
        self.assertIn('start: smiles ">>"', grammar)
        self.assertNotIn('syncode: "<<" start ">>"', grammar)

    def test_tier1_grammar_has_no_delimiters(self) -> None:
        task = get_smiles_task("acrylates")
        grammar = build_smiles_tier1_body_grammar(task["grammar_text"])
        self.assertIn("start: smiles", grammar)
        self.assertNotIn('start: smiles ">>"', grammar)
        self.assertNotIn('csd_start: smiles ">>"', grammar)

    def test_smiles_decode_caps_below_old_256(self) -> None:
        self.assertEqual(benchmark_max_new_tokens("smiles"), SMILES_TIER2_MAX_NEW_TOKENS)
        self.assertEqual(effective_max_new_tokens("smiles", 512), SMILES_TIER2_MAX_NEW_TOKENS)
        self.assertEqual(smiles_tier1_max_new_tokens(512), SMILES_TIER1_MAX_NEW_TOKENS)
        self.assertEqual(smiles_tier2_max_new_tokens(512), SMILES_TIER2_MAX_NEW_TOKENS)
        self.assertEqual(SMILES_TIER2_MAX_NEW_TOKENS, 256)
        self.assertEqual(smiles_tier1_max_new_tokens(32), 32)


if __name__ == "__main__":
    unittest.main()
