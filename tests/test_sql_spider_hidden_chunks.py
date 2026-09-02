from synthesis.evaluate.benchmarks.sql_spider import eval_logic as sql_eval_logic
from synthesis.evaluate.benchmarks.sql_spider import generation as sql_generation


def test_spider_token0_mode_uses_hidden_chunks_for_syntax_accounting():
    assert sql_eval_logic.uses_hidden_chunks() is True


def test_spider_default_prompt_is_exact_itergen_surface():
    prompt = sql_eval_logic.format_prompt(
        evaluator=object(),
        example={
            "db_id": "concert_singer",
            "db_info": "# singer ( singer_id , name )",
            "question": "How many singers do we have?",
        },
    )

    assert prompt == (
        "db_id: concert_singer\n"
        "db_info: # singer ( singer_id , name )\n"
        "question: How many singers do we have? Only output the SQL quey. \n"
        "SQL:"
    )
    assert "reason" not in prompt.lower()


def test_spider_generation_does_not_force_hidden_start(monkeypatch):
    observed = {}

    def fake_run_crane_csd(*args, **kwargs):
        observed.update(kwargs)
        return "", 0, 0.0, [], []

    monkeypatch.setattr(sql_generation, "_run_crane_csd", fake_run_crane_csd)

    result = sql_generation.run_crane_csd(
        env={},
        prompt_text="SQL: ",
        max_steps=1,
        grammar_file=None,
        start_inside_constrained=False,
    )

    assert result == ("", 0, 0.0, [], [])
    assert observed["start_inside_constrained"] is False
