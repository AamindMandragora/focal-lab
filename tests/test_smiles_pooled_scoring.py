from synthesis.evaluate.benchmarks.smiles.pooled_eval import (
    DEFAULT_SMILES_POOLED_SUCCESS_TARGET,
    SmilesPooledConfig,
    SmilesPromptFeedback,
    SmilesStopCriterion,
    aggregate_smiles_pooled_scores,
    aggregate_unique_smiles_records,
    should_stop_pooled_session,
    smiles_pooled_config_from_args,
)


def _row(
    smiles: str,
    *,
    syntax_valid: bool = True,
    in_class: bool = True,
    class_name: str = "acrylates",
) -> dict:
    return {
        "class_name": class_name,
        "extracted": smiles,
        "syntax_valid": syntax_valid,
        "smiles_eval": {
            "smiles": smiles,
            "syntax_valid": syntax_valid,
            "rdkit_available": True,
            "rdkit_valid": syntax_valid,
            "class_membership": in_class,
            "valid_class_membership": syntax_valid and in_class,
            "is_prompt_exemplar": False,
            "unique_valid_candidate": syntax_valid and in_class,
        },
    }


def test_aggregate_counts_first_occurrence_unique_only():
    target = DEFAULT_SMILES_POOLED_SUCCESS_TARGET
    rows = [
        _row("CCO"),
        _row("CCO"),
        _row("CCC", in_class=False),
        _row("CCCC"),
    ]
    summary = aggregate_unique_smiles_records(
        rows,
        success_target=target,
        prompt_exemplars=[],
    )
    assert summary.unique_syntax_valid_count == 3
    assert summary.unique_in_class_count == 2
    assert summary.syntax_rate == 3 / target
    assert summary.accuracy == 2 / target


def test_aggregate_excludes_prompt_exemplars():
    target = 10
    rows = [_row("CCO"), _row("CCC")]
    summary = aggregate_unique_smiles_records(
        rows,
        success_target=target,
        prompt_exemplars=["CCO"],
    )
    assert summary.unique_syntax_valid_count == 1
    assert summary.unique_in_class_count == 1
    assert summary.accuracy == 0.1


def test_aggregate_averages_per_class_rates():
    target = 10
    rows = [
        _row("CCO", class_name="acrylates"),
        _row("CCC", class_name="chain_extenders"),
    ]
    summary = aggregate_smiles_pooled_scores(rows, success_target=target)
    assert summary.syntax_rate == 0.1
    assert summary.accuracy == 0.1
    assert summary.unique_syntax_valid_count == 2
    assert summary.unique_in_class_count == 2


def test_should_stop_on_unique_syntax_valid_target():
    config = SmilesPooledConfig(success_target=3, max_attempts=200)
    assert not should_stop_pooled_session(
        attempt_index=0,
        config=config,
        grammar_successes=0,
        unique_syntax_valid_count=2,
    )
    assert should_stop_pooled_session(
        attempt_index=1,
        config=config,
        grammar_successes=0,
        unique_syntax_valid_count=3,
    )


def test_rs_and_cars_use_static_prompt_feedback_from_baseline_config():
    args = type(
        "Args",
        (),
        {
            "cars_search_steps": 200,
            "rs_search_steps": 200,
            "cars_success_target": DEFAULT_SMILES_POOLED_SUCCESS_TARGET,
            "smiles_unique_syntax_valid_target": DEFAULT_SMILES_POOLED_SUCCESS_TARGET,
        },
    )()
    rs_config = smiles_pooled_config_from_args(
        args,
        stop_criterion=SmilesStopCriterion.UNIQUE_SYNTAX_VALID,
        prompt_feedback=SmilesPromptFeedback.STATIC,
    )
    assert rs_config.prompt_feedback == SmilesPromptFeedback.STATIC
    assert rs_config.success_target == DEFAULT_SMILES_POOLED_SUCCESS_TARGET
