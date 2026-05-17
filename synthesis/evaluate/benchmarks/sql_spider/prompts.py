"""Spider prompt templates shared across evaluator formatting modes."""

from __future__ import annotations

from typing import Any

_SPIDER_FEW_SHOT = (
    "Example:\n"
    "db_id: concert_singer\n"
    "db_info: # singer ( singer_id , name , country , age )\n"
    "question: How many singers do we have?\n"
)


def format_spider_prompt(
    example: dict[str, Any],
    *,
    instruction: str,
    few_shot_answer_line: str,
) -> str:
    """Build a Spider task prompt from shared schema/question blocks."""
    db_id = example.get("db_id", "")
    db_info = example.get("db_info", "")
    question = example.get("question", "")
    return (
        "You are given a database schema and a question. "
        f"{instruction}\n\n"
        f"{_SPIDER_FEW_SHOT}"
        f"{few_shot_answer_line}\n\n"
        f"db_id: {db_id}\n"
        f"db_info: {db_info}\n"
        f"question: {question}\n"
        "SQL: "
    )
