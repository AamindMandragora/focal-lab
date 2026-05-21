"""Tests for << >> span extraction helpers."""

from __future__ import annotations

import unittest

from synthesis.evaluate.benchmarks.common.delimited_output import extract_sql_scored_output
from synthesis.evaluate.prompt_tiers import render_benchmark_prompt


class DelimitedOutputTests(unittest.TestCase):
    def test_extract_sql_prefers_last_delimited_span(self) -> None:
        text = "Brief plan.\n<<SELECT name FROM stadium>>"
        sql, source = extract_sql_scored_output(text)
        self.assertEqual(source, "last_visible_span")
        self.assertEqual(sql, "SELECT name FROM stadium")

    def test_extract_sql_strips_markdown_fence(self) -> None:
        text = "```sql\nSELECT COUNT(*) FROM singer\n```"
        sql, source = extract_sql_scored_output(text)
        self.assertEqual(source, "markdown_or_select_span")
        self.assertEqual(sql, "SELECT COUNT(*) FROM singer")

    def test_extract_sql_drops_trailing_fence_backticks(self) -> None:
        text = "Count rows.\nSELECT COUNT(*) FROM singer ```"
        sql, source = extract_sql_scored_output(text)
        self.assertEqual(source, "markdown_or_select_span")
        self.assertEqual(sql, "SELECT COUNT(*) FROM singer")


class SpiderPromptTests(unittest.TestCase):
    def test_tier2_spider_prompt_uses_delimiters_and_reasoning_anchor(self) -> None:
        prompt = render_benchmark_prompt(
            "spider",
            tier=2,
            example={
                "db_info": "# singer ( singer_id , name )",
                "question": "How many singers?",
            },
            max_fewshots=2,
        )
        self.assertIn("<<SELECT COUNT(*) FROM singer>>", prompt)
        self.assertTrue(prompt.rstrip().endswith("Reasoning:"))
        self.assertIn("double angle brackets", prompt.lower())


if __name__ == "__main__":
    unittest.main()
