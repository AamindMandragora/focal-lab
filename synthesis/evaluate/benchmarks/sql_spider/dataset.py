"""
Spider dataset loading utilities.

Loads text-to-SQL problems in the same shape itergen uses. Each row has:
    - db_id: database identifier (directory under the Spider databases root)
    - question: natural-language question
    - query: gold SQL (from local dev.json / dev_gold.sql)
    - db_info: schema string of the form
        "# t1 ( c1 , c2 , ... )\n# t2 ( ... )\n# t1.c1 = t2.c2\n..."
    - prompt: pre-formatted prompt body (db_id/db_info/question/SQL:)

Priority:
    1. HuggingFace `richardr1126/spider-context-validation` (matches itergen).
    2. Local fallback at $SPIDER_DATA_DIR (default: ~/spider_data/spider_data)
       using dev.json, dev_gold.sql, tables.json.
"""

from __future__ import annotations

import json
import os
import random
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from synthesis.project_defaults import default_spider_data_dir

DEFAULT_SPIDER_DIR = Path(
    default_spider_data_dir()
)
PROJECT_ROOT = Path(__file__).parent.parent.parent


def _vendored_spider_eval_dir() -> Path:
    candidates = [
        PROJECT_ROOT / "syncode" / "syncode" / "utils" / "sql_spider_eval",
        default_spider_data_dir(),
        Path.home() / "CRANE" / "src" / "crane" / "iter_syncode" / "utils" / "sql_spider_eval",
        Path.home() / "itergen" / "itergen" / "syncode" / "syncode" / "utils" / "sql_spider_eval",
    ]
    for candidate in candidates:
        tables_json = candidate / "evaluation_examples" / "examples" / "tables.json"
        databases = candidate / "databases"
        if tables_json.exists() and databases.exists():
            return candidate
    return candidates[0]


def _build_db_info_from_tables_entry(entry: Dict[str, Any]) -> str:
    """Render a tables.json entry into the itergen-style db_info string."""
    table_names = entry.get("table_names_original") or entry.get("table_names") or []
    column_names = entry.get("column_names_original") or entry.get("column_names") or []
    foreign_keys = entry.get("foreign_keys") or []

    table_to_cols: Dict[int, List[str]] = {i: [] for i in range(len(table_names))}
    for col in column_names:
        try:
            tbl_idx, col_name = col
        except ValueError:
            continue
        if tbl_idx is None or tbl_idx < 0:
            continue
        if tbl_idx in table_to_cols and col_name != "*":
            table_to_cols[tbl_idx].append(str(col_name))

    lines: List[str] = []
    for tbl_idx, tbl_name in enumerate(table_names):
        cols = table_to_cols.get(tbl_idx, [])
        if not cols:
            continue
        lines.append(f"# {tbl_name} ( {' , '.join(cols)} )")

    for fk in foreign_keys:
        try:
            src_idx, dst_idx = fk
        except (TypeError, ValueError):
            continue
        try:
            src_t, src_c = column_names[src_idx]
            dst_t, dst_c = column_names[dst_idx]
        except (IndexError, ValueError):
            continue
        if src_t < 0 or dst_t < 0:
            continue
        lines.append(
            f"# {table_names[src_t]}.{src_c} = {table_names[dst_t]}.{dst_c}"
        )

    return "\n".join(lines) + ("\n" if lines else "")


@lru_cache(maxsize=None)
def _load_tables_map(tables_path: str) -> Dict[str, Dict[str, Any]]:
    with open(tables_path) as f:
        data = json.load(f)
    return {entry["db_id"]: entry for entry in data}


def _format_prompt_body(db_id: str, db_info: str, question: str) -> str:
    """Match syncode Dataset('spider')'s prompt shape."""
    return f"db_id: {db_id}\ndb_info: {db_info}\nquestion: {question}\nSQL:"


def _load_local_spider(spider_dir: Path) -> List[Dict[str, Any]]:
    dev_path = spider_dir / "dev.json"
    gold_path = spider_dir / "dev_gold.sql"
    tables_path = spider_dir / "tables.json"
    if not (dev_path.exists() and tables_path.exists()):
        raise FileNotFoundError(
            f"Spider local files not found under {spider_dir}: "
            "expected dev.json and tables.json"
        )

    dev = json.loads(dev_path.read_text())
    tables = _load_tables_map(str(tables_path))

    # Gold SQL file is parallel to dev.json: one line per dev example, "<sql>\t<db_id>"
    gold_lines: List[str] = []
    if gold_path.exists():
        gold_lines = [
            line.rstrip("\n")
            for line in gold_path.read_text().splitlines()
            if line.strip()
        ]

    rows: List[Dict[str, Any]] = []
    for i, ex in enumerate(dev):
        db_id = ex["db_id"]
        question = ex.get("question", "")
        query = ex.get("query", "")
        gold_db_id = db_id
        if i < len(gold_lines) and "\t" in gold_lines[i]:
            gold_sql, gold_db_id = gold_lines[i].split("\t", 1)
            if gold_sql.strip():
                query = gold_sql.strip()
            gold_db_id = gold_db_id.strip()

        entry = tables.get(db_id)
        db_info = _build_db_info_from_tables_entry(entry) if entry else ""

        rows.append(
            {
                "db_id": gold_db_id or db_id,
                "question": question,
                "query": query,
                "db_info": db_info,
                "prompt": _format_prompt_body(db_id, db_info, question),
            }
        )
    return rows


def _load_hf_spider() -> List[Dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError(
            "Missing dependency `datasets`. Install with: pip install datasets"
        ) from e

    ds = load_dataset("richardr1126/spider-context-validation", split="validation")
    rows: List[Dict[str, Any]] = []
    for ex in ds:
        db_id = ex.get("db_id", "")
        db_info = ex.get("db_info", "")
        question = ex.get("question", "")
        # HF split carries "response" as the gold SQL; fall back to legacy keys.
        query = ex.get("ground_truth") or ex.get("response") or ex.get("query") or ex.get("sql") or ""
        rows.append(
            {
                "db_id": db_id,
                "question": question,
                "query": query,
                "db_info": db_info,
                "prompt": _format_prompt_body(db_id, db_info, question),
            }
        )
    return rows


def load_spider(
    source: str = "auto",
    spider_dir: Optional[Path | str] = None,
    limit: Optional[int] = None,
    random_sample: bool = False,
    seed: Optional[int] = None,
    indices: Optional[Sequence[int]] = None,
) -> List[Dict[str, Any]]:
    """
    Load the Spider dev split.

    Args:
        source: "hf", "local", or "auto" (tries HF first, then local).
        spider_dir: override local Spider directory (defaults to SPIDER_DATA_DIR env
                    or ~/spider_data/spider_data).
        limit: maximum number of rows to return.
        random_sample: if True and limit is set, pick random indices.
        seed: RNG seed when random_sample is True.
        indices: Optional explicit row indices to select before applying limit.

    Returns:
        list of dicts (see module docstring).
    """
    spider_dir = Path(spider_dir) if spider_dir is not None else DEFAULT_SPIDER_DIR

    rows: List[Dict[str, Any]] = []
    errors: List[str] = []

    if source in ("hf", "auto"):
        try:
            rows = _load_hf_spider()
            print(f"Loaded {len(rows)} Spider examples from HuggingFace")
        except Exception as e:
            errors.append(f"HF load failed: {e}")
            if source == "hf":
                raise

    if not rows and source in ("local", "auto"):
        try:
            rows = _load_local_spider(spider_dir)
            print(f"Loaded {len(rows)} Spider examples from {spider_dir}")
        except Exception as e:
            errors.append(f"local load failed: {e}")
            if source == "local":
                raise

    if not rows:
        raise RuntimeError("Failed to load Spider: " + "; ".join(errors))

    if indices is not None:
        selected: List[Dict[str, Any]] = []
        for idx in indices:
            if idx < 0 or idx >= len(rows):
                raise IndexError(
                    f"Spider index {idx} is out of range for {len(rows)} loaded rows"
                )
            row = dict(rows[idx])
            row["spider_source_index"] = idx
            selected.append(row)
        rows = selected

    if limit is not None and limit > 0:
        if random_sample:
            rng = random.Random(seed) if seed is not None else random
            indices = rng.sample(range(len(rows)), min(limit, len(rows)))
            rows = [rows[i] for i in indices]
        else:
            rows = rows[: min(limit, len(rows))]

    return rows


def make_spider_train_test_split(
    total_examples: int,
    train_size: int = 50,
    test_size: int = 100,
    seed: int = 123,
) -> Dict[str, Any]:
    """Create a deterministic non-overlapping Spider train/test index split."""
    if total_examples <= 0:
        raise ValueError("total_examples must be positive")
    if train_size <= 0 or test_size <= 0:
        raise ValueError("train_size and test_size must be positive")
    if train_size + test_size > total_examples:
        raise ValueError(
            f"Requested train_size + test_size = {train_size + test_size}, "
            f"but only {total_examples} Spider examples are available"
        )

    rng = random.Random(seed)
    shuffled = list(range(total_examples))
    rng.shuffle(shuffled)
    train_indices = sorted(shuffled[:train_size])
    test_indices = sorted(shuffled[train_size: train_size + test_size])
    return {
        "seed": seed,
        "split_strategy": "random_non_overlapping",
        "total_examples": total_examples,
        "train_indices": train_indices,
        "test_indices": test_indices,
        "train_size": len(train_indices),
        "test_size": len(test_indices),
    }


def write_spider_train_test_split(
    output_path: Path | str,
    *,
    source: str = "auto",
    spider_dir: Optional[Path | str] = None,
    train_size: int = 50,
    test_size: int = 100,
    seed: int = 123,
    include_preview: bool = True,
) -> Dict[str, Any]:
    """Write a deterministic Spider split manifest for synthesis/test workflows."""
    rows = load_spider(source=source, spider_dir=spider_dir)
    split = make_spider_train_test_split(
        total_examples=len(rows),
        train_size=train_size,
        test_size=test_size,
        seed=seed,
    )
    if include_preview:
        for split_name in ("train", "test"):
            previews = []
            for idx in split[f"{split_name}_indices"][:10]:
                row = rows[idx]
                previews.append({
                    "index": idx,
                    "db_id": row.get("db_id", ""),
                    "question": row.get("question", ""),
                })
            split[f"{split_name}_preview"] = previews

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(split, indent=2))
    return split


def write_gold_file(examples: List[Dict[str, Any]], path: Path) -> None:
    """Write a Spider-format gold file (<sql>\\t<db_id>) for the given examples."""
    with open(path, "w") as f:
        for ex in examples:
            gold = (ex.get("query") or "").replace("\n", " ").strip()
            db_id = ex.get("db_id", "")
            f.write(f"{gold}\t{db_id}\n")


def default_db_dir() -> Path:
    """Default SQLite databases directory for Spider."""
    env = os.environ.get("SPIDER_DB_DIR")
    if env:
        return Path(env)
    local = DEFAULT_SPIDER_DIR / "database"
    if local.exists():
        return local
    return _vendored_spider_eval_dir() / "databases"


def default_tables_json() -> Path:
    """Default tables.json path for Spider."""
    env = os.environ.get("SPIDER_TABLES_JSON")
    if env:
        return Path(env)
    local = DEFAULT_SPIDER_DIR / "tables.json"
    if local.exists():
        return local
    return _vendored_spider_eval_dir() / "evaluation_examples" / "examples" / "tables.json"
