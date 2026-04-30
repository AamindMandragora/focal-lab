from evaluation.evaluator import Evaluator


def test_spider_extra_tokens_include_quoted_question_literals():
    evaluator = Evaluator(dataset_name="spider")
    dataset = [
        {
            "question": "what is the biggest city in wyoming",
            "schema": "city: city_name (text), state_name (text), population (number)",
        }
    ]

    tokens = set(evaluator._collect_spider_extra_token_strings(dataset))

    assert '"' in tokens
    assert "'" in tokens
    assert '"wyoming"' in tokens
    assert ' "wyoming"' in tokens


def test_spider_extra_tokens_optionally_include_full_prompt(monkeypatch):
    monkeypatch.setenv("CSD_SPIDER_INCLUDE_PROMPT_TOKENS", "1")
    evaluator = Evaluator(dataset_name="spider")
    dataset = [
        {
            "question": "what is the biggest city in wyoming",
            "schema": "city: city_name (text), state_name (text), population (number)",
            "db_id": "geo",
        }
    ]

    tokens = set(evaluator._collect_spider_extra_token_strings(dataset))

    # Full prompt should be included as an extra string when enabled.
    assert any("You are a text-to-SQL system for the Spider benchmark." in t for t in tokens)
