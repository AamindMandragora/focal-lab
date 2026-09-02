"""Trace summaries must not serialize private logits or generated token text."""

import importlib.util
import pathlib


_REPO = pathlib.Path(__file__).resolve().parents[3]
_ENVIRONMENT_PATH = (
    _REPO / "synthesis" / "evaluate" / "benchmarks" / "gsm_symbolic" / "environment.py"
)


def _load_environment():
    spec = importlib.util.spec_from_file_location("_gsm_environment_trace_test", _ENVIRONMENT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_secret_absent(event, *secrets):
    rendered = str(event)
    for secret in secrets:
        assert secret not in rendered


def test_snapshot_trace_records_shape_without_logit_values():
    env = _load_environment()
    event = env._summarize_helper_event(
        "SaveLogitsSnapshot", (object(),), [1234.567, -7654.321, 42.0], 8, 8
    )

    assert event["snapshot_size"] == 3
    assert event["detail"] == "saved logits snapshot, size=3"
    _assert_secret_absent(event, "1234.567", "-7654.321", "42.0")


def test_restore_trace_records_shape_without_logit_values():
    env = _load_environment()
    snapshot = [1234.567, -7654.321]
    event = env._summarize_helper_event(
        "RestoreLogitsSnapshot", (object(), snapshot), None, 8, 8
    )

    assert event["snapshot_size"] == 2
    assert event["detail"] == "restored logits snapshot, size=2"
    _assert_secret_absent(event, "1234.567", "-7654.321")


def test_speculative_trace_records_counts_and_flags_without_candidate_text():
    env = _load_environment()
    result = (["PRIVATE_CANDIDATE"], ["PRIVATE_PREFIX", "PRIVATE_CANDIDATE"], True, False, 1)
    event = env._summarize_helper_event(
        "SpeculativeConstrainedRollout", (object(),), result, 11, 12
    )

    assert event["candidate_token_count"] == 1
    assert event["candidate_prefix_len"] == 2
    assert event["hit_complete"] is True
    assert event["hit_eos"] is False
    assert event["steps_used"] == 1
    _assert_secret_absent(event, "PRIVATE_CANDIDATE", "PRIVATE_PREFIX")


def test_rollback_continue_trace_records_lengths_without_generated_text():
    env = _load_environment()
    result = (["PRIVATE_GENERATED"], ["PRIVATE_CURRENT"])
    event = env._summarize_helper_event("RollbackAndContinue", (object(),), result, 4, 6)

    assert event["generated_len"] == 1
    assert event["current_len"] == 1
    _assert_secret_absent(event, "PRIVATE_GENERATED", "PRIVATE_CURRENT")


def test_generation_helpers_record_lengths_without_generated_text():
    env = _load_environment()
    cases = (
        ("UnconstrainedGeneration", ["PRIVATE_FREE"], 1, None),
        ("ConstrainedGeneration", (["PRIVATE_CONSTRAINED"], True), 1, True),
        ("CraneGeneration", ["PRIVATE_CRANE"], 1, None),
    )

    for name, result, expected_len, expected_eos in cases:
        event = env._summarize_helper_event(name, (object(),), result, 1, 2)
        assert event["generated_len"] == expected_len
        if expected_eos is not None:
            assert event["terminated_by_eos"] is expected_eos
        _assert_secret_absent(event, "PRIVATE_FREE", "PRIVATE_CONSTRAINED", "PRIVATE_CRANE")


def test_penalized_rollout_trace_records_status_without_generated_text():
    env = _load_environment()
    result = (["PRIVATE_ROLLOUT"], 3, True)
    event = env._summarize_helper_event(
        "RolloutConstrainedWithPenalties", (object(),), result, 2, 5
    )

    assert event["generated_len"] == 1
    assert event["steps_used"] == 3
    assert event["terminated_by_eos"] is True
    _assert_secret_absent(event, "PRIVATE_ROLLOUT")


def test_unrecognized_helper_fallback_never_serializes_result():
    env = _load_environment()
    event = env._summarize_helper_event(
        "FuturePublicHelper", (object(),), "PRIVATE_FUTURE_RESULT", 0, 0
    )

    assert event["detail"] == "result redacted"
    _assert_secret_absent(event, "PRIVATE_FUTURE_RESULT")
