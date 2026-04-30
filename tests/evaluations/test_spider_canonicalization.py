from evaluation.evaluator import Evaluator


def _mk_example(question: str) -> dict:
    return {
        "question": question,
        "query": (
            'SELECT city_name FROM city WHERE population = '
            '( SELECT MAX ( population ) FROM city WHERE state_name = "wyoming" ) '
            'AND state_name = "wyoming"'
        ),
    }


def test_spider_canonicalize_state_only_superlative_question():
    evaluator = Evaluator(dataset_name="spider")
    example = _mk_example("which city in wyoming has the largest population")
    sql = "select city_name from city where state_name='wyoming'"

    got = evaluator._canonicalize_spider_sql(sql, example)

    assert "max" in got.lower()
    assert "population" in got.lower()
    assert '"wyoming"' in got.lower()


def test_spider_canonicalize_city_only_superlative_question():
    evaluator = Evaluator(dataset_name="spider")
    example = _mk_example("where is the most populated area of wyoming")
    sql = "select city_name from city"

    got = evaluator._canonicalize_spider_sql(sql, example)

    assert "max" in got.lower()
    assert "population" in got.lower()
    assert "state_name" in got.lower()
