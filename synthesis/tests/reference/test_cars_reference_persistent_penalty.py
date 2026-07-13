from pathlib import Path


def test_cars_reference_records_failed_token_at_failed_prefix():
    repo = Path(__file__).resolve().parents[2]
    body = (repo / "verify" / "reference" / "cars.dfy").read_text()

    expected = "lm.PenalizeTriedTokenAt(prompt + cur, next);"
    assert expected in body, (
        "CARS reconstruction must persistently penalize the failed token at the "
        "current prefix before rollback, so CARS-style full-vocabulary retries "
        "can use the recorded failure."
    )
