import json
from pathlib import Path

from synthesis.evaluate.benchmarks.sql_spider.dataset import load_spider
from synthesis.evaluate.benchmarks.sql_spider.executor import (
    ensure_spider_nltk_prereqs,
    execute_accuracy,
    prediction_matches_gold,
)


def test_spider_execution_scoring_works_after_nltk_prereqs():
    ensure_spider_nltk_prereqs()

    split = json.loads(
        Path("environment/benchmark_splits/spider_dev_proportional.json").read_text()
    )
    examples = load_spider(
        source="auto",
        limit=3,
        random_sample=False,
        indices=split["test_indices"][:3],
    )
    example = examples[0]

    assert prediction_matches_gold("SELECT COUNT(*) FROM singer", example)

    oracle_preds = [(ex.get("query") or "").strip() for ex in examples]
    _, _, per_row = execute_accuracy(oracle_preds, examples)
    assert all(row.get("exec") for row in per_row)
