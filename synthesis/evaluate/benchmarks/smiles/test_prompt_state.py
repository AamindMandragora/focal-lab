"""Unit tests for SMILES prompt accumulation."""

from __future__ import annotations

import unittest

from synthesis.evaluate.benchmarks.smiles.prompt_state import SmilesPromptState, record_prompt_result


class SmilesPromptStateTests(unittest.TestCase):
    def test_records_good_and_bad_sections(self) -> None:
        state = SmilesPromptState(["CCO"])
        example = {"class_name": "chain_extenders", "prompt": "Task text\nReasoning:\n"}

        good_row = {
            "unique_valid_candidate": True,
            "is_prompt_exemplar": False,
        }
        bad_row = {
            "unique_valid_candidate": False,
            "syntax_valid": False,
            "is_prompt_exemplar": False,
        }

        record_prompt_result(example, {"chain_extenders": state}, "OCCO", good_row)
        record_prompt_result(example, {"chain_extenders": state}, "BAD!", bad_row)
        state.apply_to_example(example)

        self.assertIn("Good results:", example["prompt"])
        self.assertIn("SMILES: OCCO", example["prompt"])
        self.assertIn("Bad results:", example["prompt"])
        self.assertIn("SMILES: BAD!", example["prompt"])
        self.assertNotIn("SMILES: CCO", example["prompt"].split("Good results:")[-1])

    def test_exemplar_copy_is_appended_to_bad_results(self) -> None:
        state = SmilesPromptState(["CC(=C)C(=O)OCC1=CC=CC=C1"])
        example = {"class_name": "acrylates", "prompt": "Task\nReasoning:\n"}
        row = {
            "unique_valid_candidate": False,
            "syntax_valid": True,
            "is_prompt_exemplar": True,
        }

        outcome = record_prompt_result(
            example,
            {"acrylates": state},
            "CC(=C)C(=O)OCC1=CC=CC=C1",
            row,
        )
        state.apply_to_example(example)

        self.assertEqual(outcome and outcome.get("prompt_record_outcome"), "exemplar")
        self.assertIn("Bad results:", example["prompt"])
        self.assertIn("SMILES: CC(=C)C(=O)OCC1=CC=CC=C1", example["prompt"])

    def test_syntax_invalid_is_appended_to_bad_results(self) -> None:
        state = SmilesPromptState([])
        example = {"class_name": "acrylates", "prompt": "Task\nReasoning:\n"}
        row = {
            "unique_valid_candidate": False,
            "syntax_valid": False,
            "is_prompt_exemplar": False,
        }

        record_prompt_result(example, {"acrylates": state}, "not-a-smiles", row)
        state.apply_to_example(example)

        self.assertIn("Bad results:", example["prompt"])
        self.assertIn("SMILES: not-a-smiles", example["prompt"])

    def test_duplicate_valid_is_not_counted_as_good(self) -> None:
        state = SmilesPromptState([])
        states = {"acrylates": state}
        row = {"unique_valid_candidate": True, "is_prompt_exemplar": False}

        first = record_prompt_result({"class_name": "acrylates"}, states, "C=CC(=O)O", row)
        second = record_prompt_result({"class_name": "acrylates"}, states, "C=CC(=O)O", row)

        self.assertTrue(first and first["novel_valid"])
        self.assertFalse(second and second["novel_valid"])
        self.assertEqual(state.good_results, ["C=CC(=O)O"])


if __name__ == "__main__":
    unittest.main()
