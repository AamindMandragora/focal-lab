from pathlib import Path

import pytest

from synthesis.evaluate.baselines.crane_repo_runner import (
    _latest_crane_results,
    _load_expected_crane_rows,
)


def test_latest_crane_results_is_scoped_to_the_requested_run(tmp_path: Path) -> None:
    logging_root = tmp_path / "logging" / "gsm_symbolic" / "no_judge"
    requested = (
        logging_root
        / "cot-model=Qwen3.5-4B"
        / "cot=False"
        / "parsing=regex"
        / "gsm-gsm"
        / "cot-grammar-mode=itergen"
        / "8-shot_1_samples_True.jsonl"
    )
    competing = (
        logging_root
        / "cot-model=Qwen3.5-4B"
        / "cot=True"
        / "parsing=regex"
        / "text-text"
        / "cot-grammar-mode=original"
        / "8-shot_1_samples_True.jsonl"
    )
    requested.parent.mkdir(parents=True)
    competing.parent.mkdir(parents=True)
    requested.write_text('{"requested": true}\n')
    competing.write_text('{"competing": true}\n')
    competing.touch()

    selected = _latest_crane_results(
        tmp_path,
        "gsm_symbolic",
        eval_model="Qwen/Qwen3.5-4B",
        mode="itergen",
        do_cot=False,
        grammar_flag="gsm",
    )

    assert selected == requested


def test_load_expected_crane_rows_rejects_incomplete_result(tmp_path: Path) -> None:
    result = tmp_path / "result.jsonl"
    result.write_text('{"row": 1}\n{"row": 2}\n')

    with pytest.raises(
        RuntimeError,
        match=r"gcd/gsm_symbolic: expected 49, found 2",
    ):
        _load_expected_crane_rows(
            result,
            expected_rows=49,
            strategy="gcd",
            dataset="gsm_symbolic",
        )
