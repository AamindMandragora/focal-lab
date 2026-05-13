"""Spider evaluation logic delegated from the global evaluator."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


def get_grammar_file(evaluator: Any, grammars_dir: Path) -> Path:
    return grammars_dir / "sql.lark"


def load_dataset_sample(evaluator: Any) -> list[dict[str, Any]]:
    from synthesis.evaluate.benchmarks.sql_spider.dataset import load_spider

    split_indices = evaluator._load_spider_split_indices()
    ds = load_spider(
        source="auto",
        limit=evaluator.sample_size,
        random_sample=split_indices is None,
        seed=evaluator.sample_seed,
        indices=split_indices,
    )
    return list(ds)


def format_prompt(evaluator: Any, example: dict[str, Any]) -> str:
    db_id = example.get("db_id", "")
    db_info = example.get("db_info", "")
    question = example.get("question", "")
    return (
        "You are given a database schema and a question. "
        "Write a SINGLE SQL query answering the question, using ONLY the tables and columns in the schema.\n\n"
        "You may optionally reason about the problem first. "
        "Then, wrap your final SQL query in << >> delimiters. "
        "Stop after the closing >>.\n\n"
        "Example:\n"
        "db_id: concert_singer\n"
        "db_info: # singer ( singer_id , name , country , age )\n"
        "question: How many singers do we have?\n"
        "SQL: <<SELECT count(*) FROM singer>>\n\n"
        f"db_id: {db_id}\n"
        f"db_info: {db_info}\n"
        f"question: {question}\n"
        "SQL: "
    )


def format_prompt_expression_only(evaluator: Any, example: dict[str, Any]) -> str:
    """Hard-mask / constrained decoders: emit only ``SQL: <<query>>``."""
    db_id = example.get("db_id", "")
    db_info = example.get("db_info", "")
    question = example.get("question", "")
    return (
        "You are given a database schema and a question. "
        "Write ONE SQL query using ONLY tables and columns shown in the schema.\n\n"
        "Output a single line of the form SQL: <<YOUR QUERY>> — no reasoning or other text.\n\n"
        "Example:\n"
        "db_id: concert_singer\n"
        "db_info: # singer ( singer_id , name , country , age )\n"
        "question: How many singers do we have?\n"
        "SQL: <<SELECT count(*) FROM singer>>\n\n"
        f"db_id: {db_id}\n"
        f"db_info: {db_info}\n"
        f"question: {question}\n"
        "SQL: "
    )


def format_prompt_chain_of_thought(evaluator: Any, example: dict[str, Any]) -> str:
    """Legacy CRANE-style runs: require explicit reasoning before the delimited query."""
    db_id = example.get("db_id", "")
    db_info = example.get("db_info", "")
    question = example.get("question", "")
    return (
        "You are given a database schema and a question. "
        "Write a SINGLE SQL query answering the question, using ONLY the tables and columns in the schema.\n\n"
        "Reason step by step (tables, joins, filters). "
        "Then output SQL: followed by your query wrapped in << >>. "
        "Stop after the closing >>.\n\n"
        "Example:\n"
        "db_id: concert_singer\n"
        "db_info: # singer ( singer_id , name , country , age )\n"
        "question: How many singers do we have?\n"
        "Let's think step by step. We only need the singer table. "
        "SQL: <<SELECT count(*) FROM singer>>\n\n"
        f"db_id: {db_id}\n"
        f"db_info: {db_info}\n"
        f"question: {question}\n"
        "SQL: "
    )


def expected_answer(evaluator: Any, example: dict[str, Any]) -> str:
    return (example.get("query") or "").strip()


def build_dynamic_parser(evaluator: Any, env: dict[str, Any], example: dict[str, Any]):
    return None


def extract_actual(evaluator: Any, scored_output: str, example: dict[str, Any]) -> tuple[str | None, str, dict[str, Any] | None]:
    import re
    if not scored_output:
        return None, "none", None
    # Prefer extracting from << >> delimiters (CRANE, IterGen, CARS, GCD).
    expr_matches = re.findall(r"<<\s*([^<>]+?)\s*>>", scored_output)
    if expr_matches:
        cleaned = expr_matches[-1].replace("\n", " ").replace("\r", " ").strip()
        cleaned = " ".join(cleaned.split()).rstrip(";").strip()
        return (cleaned or None), "last_visible_span", None
    # Fallback: treat the raw first paragraph as the query (e.g. unconstrained).
    raw = scored_output.split("\n\n")[0]
    cleaned = raw.replace("\n", " ").replace("\r", " ").strip()
    cleaned = cleaned.replace("<<", " ").replace(">>", " ")
    cleaned = " ".join(cleaned.split()).rstrip(";").strip()
    return (cleaned or None), ("raw_text_fallback" if cleaned else "none"), None


def is_correct(
    evaluator: Any,
    actual: str | None,
    expected: str,
    example: dict[str, Any],
    aux: dict[str, Any] | None,
    scored_output: str,
) -> bool:
    if not actual or not expected:
        return False
    from synthesis.evaluate.benchmarks.sql_spider.dataset import default_db_dir

    db_id = example.get("db_id", "")
    if not db_id:
        return False
    db_path = default_db_dir() / db_id / f"{db_id}.sqlite"
    if not db_path.exists():
        return False

    def _run(sql: str):
        try:
            con = sqlite3.connect(str(db_path))
            con.text_factory = lambda b: b.decode("utf-8", errors="ignore")
            cur = con.cursor()
            cur.execute(sql)
            rows = cur.fetchall()
            con.close()
            return rows
        except Exception:
            return None

    pred_rows = _run(actual)
    gold_rows = _run(expected)
    if pred_rows is None or gold_rows is None:
        return False
    try:
        return sorted(map(tuple, pred_rows)) == sorted(map(tuple, gold_rows))
    except TypeError:
        return list(map(tuple, pred_rows)) == list(map(tuple, gold_rows))


def uses_hidden_chunks() -> bool:
    return False


def example_syntax_pass(
    all_valid_syntax: bool,
    segments: list[tuple[str, bool]],
    used_hidden_chunk: bool,
    aux: dict[str, Any] | None,
) -> bool:
    return bool(segments) and all_valid_syntax


def accuracy_applicable(aux: dict[str, Any] | None) -> bool:
    return True


def get_generation_runner():
    from synthesis.evaluate.benchmarks.sql_spider.generation import run_crane_csd

    return run_crane_csd


def get_syntax_parser(evaluator: Any, example: dict[str, Any] | None):
    from lark import Lark

    return Lark(evaluator._get_grammar_text(), start="start", parser="lalr")


def ensure_runtime_prereqs(evaluator: Any) -> None:
    return None


def compute_aux_metrics(evaluator: Any, sample_outputs: list[dict[str, Any]]) -> dict[str, Any]:
    return {}


def accuracy_upper_bound(
    num_correct: int,
    remaining: int,
    num_accuracy_examples: int,
    total_planned_examples: int,
) -> float:
    return (num_correct + remaining) / max(1, total_planned_examples)


def final_accuracy_denominator(num_examples: int, num_accuracy_examples: int) -> int:
    return num_examples


def invalid_outputs_excluded(num_examples: int, num_accuracy_examples: int) -> int:
    return 0


def accuracy_definition() -> str:
    return "correct_examples_over_all_examples"
