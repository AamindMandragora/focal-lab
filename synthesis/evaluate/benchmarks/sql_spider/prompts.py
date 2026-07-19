"""Spider prompt templates shared across evaluator formatting modes."""

from __future__ import annotations

from typing import Any

from synthesis.evaluate.benchmarks.prompt_profiles import render_strategy_prompt

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


def format_spider_itergen_aligned_prompt(example: dict[str, Any]) -> str:
    """IterGen's EXACT Spider prompt for fair head-to-head comparison.

    IterGen feeds instruct models a single user turn whose content is built as
    ``f"db_id: {db_id}\\ndb_info: {db_info}\\nquestion: {question}"`` + the
    literal suffix ``" Only output the SQL quey. \\nSQL:"`` (the ``quey`` typo
    and the leading space are IterGen's own -- preserved verbatim so the token
    sequence matches byte-for-byte). The evaluator wraps this returned string in
    ``[{"role": "user", "content": <this>}]`` and applies the tokenizer chat
    template with ``add_generation_prompt=True`` -- the same call IterGen makes,
    so the final token sequence is identical.

    No few-shot example and no ``<< >>`` instruction: span opening must come from
    a FORCING strategy (OpenConstrainedSpan), not from the model emitting ``<<``.
    """
    return render_strategy_prompt("spider", "itergen", example)


def format_spider_messages(
    example: dict[str, Any],
    *,
    instruction: str,
    few_shot_answer_line: str,
) -> list[dict]:
    """Multi-turn chat delivery of the same Spider prompt.

    The single inline few-shot example is delivered as a user/assistant turn
    pair instead of being flattened into one user message. On Spider-1.5B
    unconstrained this lifted accuracy 38.0% -> 44.0% (+6pp) over the flattened
    form with zero content change. Mirrors the GSM multi-turn fix.
    """
    db_id = example.get("db_id", "")
    db_info = example.get("db_info", "")
    question = example.get("question", "")
    # The few-shot example schema/question as a clean user turn (strip the
    # "Example:\n" label and trailing newline used only in the flattened form).
    example_user = _SPIDER_FEW_SHOT.removeprefix("Example:\n").rstrip("\n")
    real_user = f"db_id: {db_id}\ndb_info: {db_info}\nquestion: {question}"
    return [
        {"role": "system", "content": "You are given a database schema and a question. " + instruction},
        {"role": "user", "content": example_user},
        {"role": "assistant", "content": few_shot_answer_line},
        {"role": "user", "content": real_user},
    ]
